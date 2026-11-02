import json
import time
import threading

import requests
from rag.generation.cancellation import CancellationToken, call_cancellable, iter_cancellable
from rag.generation.budget import ensure_request_fits
from rag.generation.errors import ContextLengthError, LLMError


def is_context_length_error(data):
    """Only explicit provider context errors qualify; never echo their payload."""
    error = data.get("error") if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return False
    codes = {str(error.get(key, "")).lower() for key in ("code", "type")}
    if codes & {"context_length_exceeded", "max_context_length_exceeded", "context_window_exceeded"}:
        return True
    import re
    message = error.get("message", "")
    return isinstance(message, str) and bool(re.search(
        r"(?:maximum|max) context length|context (?:length|window).{0,80}(?:exceed|too (?:long|large))|"
        r"(?:input|prompt|messages).{0,80}(?:exceeds? .{0,20}(?:context|token)|too (?:long|large))|"
        r"上下文.{0,20}(?:超出|超过|过长)", message, re.I))


class DeepSeekClient:
    """参考用户的 requests 调用方式，补充流式读取、超时及错误处理。"""
    def __init__(self, settings):
        self.settings = settings

    def payload(self, messages, stream, max_tokens=None, temperature=None):
        ensure_request_fits(self.settings, messages, max_tokens)
        return {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "max_tokens": self.settings.max_output_tokens if max_tokens is None else max_tokens,
            "thinking": {"type": "disabled"},
            "stream": stream,
        }

    def _open(self, messages, stream, max_tokens=None, temperature=None, cancel_token=None):
        if not self.settings.api_key:
            raise LLMError("请在 config.py 的 DEEPSEEK_API_KEY 中填写 API key，保存并重启服务。")
        payload = self.payload(messages, stream, max_tokens, temperature)
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + self.settings.api_key}
        for attempt in range(3):
            if cancel_token:
                cancel_token.check()
            try:
                response = call_cancellable(lambda: requests.post(url, headers=headers,
                                         json=payload,
                                         timeout=(10, self.settings.timeout), stream=stream or bool(cancel_token)),
                                            cancel_token, response=True)
            except requests.ConnectTimeout:
                if cancel_token:
                    cancel_token.check()
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                raise LLMError("无法连接 DeepSeek，请检查网络后重试。") from None
            except requests.ReadTimeout:
                if cancel_token:
                    cancel_token.check()
                raise LLMError("等待 DeepSeek 响应超时；请求可能已被处理，请稍后手动重试。") from None
            except requests.RequestException:
                if cancel_token:
                    cancel_token.check()
                raise LLMError("DeepSeek 连接失败或已中断，请检查网络后手动重试。") from None
            if cancel_token and cancel_token.event.is_set():
                response.close()
                cancel_token.check()
            if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                response.close()
                time.sleep(0.5 * (attempt + 1))
                continue
            if not response.ok:
                status = response.status_code
                try:
                    if status in (400, 413, 422):
                        if cancel_token:
                            cancel_token.add_response(response)
                        try:
                            data = call_cancellable(response.json, cancel_token)
                        except (ValueError, requests.RequestException):
                            data = None
                        if is_context_length_error(data):
                            raise ContextLengthError("模型上下文长度超限，请缩小问题范围或分步提问。")
                finally:
                    self._close_response(response, cancel_token)
                known = {401: "DeepSeek API key 无效，请检查配置。",
                         402: "DeepSeek 账户余额不足。", 403: "DeepSeek 拒绝了该请求。",
                         429: "DeepSeek 请求过于频繁，请稍后重试。"}
                raise LLMError(known.get(status, f"DeepSeek 服务返回 HTTP {status}，请稍后重试。"))
            return response
        raise LLMError("DeepSeek 服务暂时不可用。")

    def complete(self, messages, max_tokens=None, temperature=None, cancel_token=None):
        # 可取消请求先只打开响应头，再读取正文，便于关闭连接。
        response = self._open(messages, False, max_tokens, temperature, cancel_token)
        try:
            if cancel_token:
                cancel_token.add_response(response)
            try:
                data = call_cancellable(response.json, cancel_token)
                answer = data["choices"][0]["message"]["content"]
                finish = data["choices"][0].get("finish_reason")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError()
            except (ValueError, KeyError, IndexError, TypeError, requests.RequestException):
                if cancel_token:
                    cancel_token.check()
                raise LLMError("DeepSeek 返回了无法解析或为空的回答。") from None
            finally:
                if cancel_token:
                    cancel_token.remove_response(response)
            if cancel_token:
                cancel_token.check()
            if finish not in (None, "stop", "length"):
                raise LLMError("DeepSeek 未正常完成回答，请稍后重试。")
            if finish == "length":
                answer += "\n\n> 本次回答达到生成长度上限，可缩小问题范围后继续提问。"
            return answer
        finally:
            self._close_response(response, cancel_token)

    @staticmethod
    def _close_response(response, cancel_token):
        if cancel_token:
            cancel_token.remove_response(response)
            threading.Thread(target=CancellationToken._close, args=(response,), daemon=True).start()
        else:
            response.close()

    def stream(self, messages, cancel_token=None):
        response = self._open(messages, True, cancel_token=cancel_token)
        response.encoding = "utf-8"
        received, finished = False, False
        lines = iter_cancellable(response.iter_lines(chunk_size=1024, decode_unicode=True), cancel_token)
        try:
            if cancel_token:
                cancel_token.add_response(response)
            for line in lines:
                if cancel_token:
                    cancel_token.check()
                if not line or not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    finished = True
                    break
                try:
                    data = json.loads(value)
                except json.JSONDecodeError:
                    raise LLMError("DeepSeek 返回的流数据格式不正确，请重试。") from None
                if not isinstance(data, dict):
                    raise LLMError("DeepSeek 返回的流数据结构不正确，请重试。")
                if data.get("error"):
                    if not received and is_context_length_error(data):
                        raise ContextLengthError("模型上下文长度超限，请缩小问题范围或分步提问。")
                    raise LLMError("DeepSeek 生成过程中发生错误，请重试。")
                choices = data.get("choices") or []
                if not isinstance(choices, list):
                    raise LLMError("DeepSeek 返回的流数据结构不正确，请重试。")
                if not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict) or not isinstance(choice.get("delta") or {}, dict):
                    raise LLMError("DeepSeek 返回的流数据结构不正确，请重试。")
                if choice.get("finish_reason") not in (None, "stop", "length"):
                    raise LLMError("DeepSeek 未正常完成回答，已收到的内容可能不完整，请重试。")
                text = (choice.get("delta") or {}).get("content")
                if text is not None and not isinstance(text, str):
                    raise LLMError("DeepSeek 返回的回答片段格式不正确，请重试。")
                if text:
                    received = True
                    yield text
                if choice.get("finish_reason") == "length":
                    yield "\n\n> 本次回答达到生成长度上限，可缩小问题范围后继续提问。"
                if choice.get("finish_reason") is not None:
                    finished = True
            if cancel_token:
                cancel_token.check()
            if not finished:
                raise LLMError("DeepSeek 连接提前结束，已收到的内容可能不完整，请重试。")
            if not received:
                raise LLMError("DeepSeek 未返回回答内容，请重试。")
        except requests.RequestException:
            if cancel_token:
                cancel_token.check()
            raise LLMError("DeepSeek 响应中断，已收到的内容可能不完整，请重试。") from None
        finally:
            lines.close()
            self._close_response(response, cancel_token)
