"use strict";
const $ = (id) => document.getElementById(id);
let busy = false;
let activeConversation = null;
let olderBefore = null;
const isGuest = document.querySelector('meta[name="user-mode"]')?.content === "guest";
const conversationKey = `lingxing-conversation-${document.querySelector('meta[name="user-id"]').content}`;
async function secureFetch(url, options = {}) {
  options.headers = {...(options.headers || {}), "X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content};
  const response = await fetch(url, options);
  if (response.status === 401) { location.assign("/login"); throw new Error("登录已过期，请重新登录。"); }
  return response;
}
async function jsonRequest(url, options = {}) {
  const response = await secureFetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败，请重试。");
  return data;
}
let currentStatus = null;
let statusTimer = null;

function showError(message) { $("error").textContent = message; $("error").classList.remove("hidden"); }
function setBusy(value) {
  busy = value;
  $("send").disabled = value;
  $("clear-chat").disabled = value;
  $("category").disabled = value;
  $("new-conversation").disabled = value;
  $("older-turns").disabled = value;
  $("logout").disabled = value;
  document.querySelectorAll(".conversation-choice").forEach(button => { button.disabled = value; });
  $("retrieve-only").disabled = value;
  document.querySelectorAll("[data-question]").forEach((button) => { button.disabled = value; });
}
function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

async function loadStatus() {
  try {
    const response = await secureFetch("/api/status");
    if (!response.ok) throw new Error("无法读取服务状态");
    const status = await response.json();
    currentStatus = status;
    $("index-label").textContent = status.index_ready ? `${status.document_count} 篇文档已就绪` : "文档库等待建立索引";
    $("index-detail").textContent = status.job.running ? status.job.message : `${status.chunk_count} 个可检索片段 · ${status.rerank_enabled ? "已启用本地重排序" : "本地知识库"}`;
    $("update-index").disabled = isGuest || status.job.running;
    if (!status.llm_configured) {
      $("config-notice").textContent = "填写 config.py 中的 DEEPSEEK_API_KEY 并重启后，即可生成回答。当前可使用检索预览。";
      $("config-notice").classList.remove("hidden");
    } else { $("config-notice").classList.add("hidden"); }
    const selected = $("category").value;
    const optionValues = [...$("category").options].map((option) => option.value).slice(1);
    if (JSON.stringify(optionValues) !== JSON.stringify(status.categories)) {
      $("category").replaceChildren(new Option("全部业务", ""), ...status.categories.map((name) => new Option(name, name)));
      if (status.categories.includes(selected)) $("category").value = selected;
    }
    if (status.job.error) showError(status.job.error);
    if (status.job.running && !statusTimer) {
      statusTimer = setInterval(loadStatus, 1600);
    } else if (!status.job.running && statusTimer) {
      clearInterval(statusTimer); statusTimer = null;
    }
  } catch (error) { $("index-label").textContent = "服务连接失败"; showError(error.message); }
}

function showSources(container, sources, turnId) {
  const details = node("details", "sources");
  details.append(node("summary", "", `依据与参考 · ${sources.length} 处来源`));
  for (const source of sources) {
    const item = node("div", "source-item");
    item.id = `${turnId}-${source.id}`;
    const heading = node("div", "source-heading");
    const title = node("span");
    title.append(node("span", "source-id", `[${source.id}] `), node("span", "", source.title));
    heading.append(title);
    if (source.document_id) {
      const localLink = node("a", "", "查看文档 ↗");
      localLink.href = `/documents/${encodeURIComponent(source.document_id)}`;
      localLink.target = "_blank"; localLink.rel = "noopener noreferrer";
      heading.append(localLink);
    }
    item.append(heading);
    if (source.source_type === "liked_qa") item.append(node("span", "source-badge", "点赞问答 · 辅助参考"));
    item.append(node("p", "", [source.category, source.heading].filter(Boolean).join(" / ")));
    if (source.api_path) item.append(node("p", "", `${source.method} ${source.api_path}`));
    const excerpt = node("details", "source-excerpt");
    excerpt.append(node("summary", "", "检索片段"), node("pre", "", source.excerpt));
    item.append(excerpt);
    if (source.source_url) {
      try {
        const url = new URL(source.source_url);
        if (["https:", "http:"].includes(url.protocol)) {
          const original = node("a", "", "官方原文 ↗");
          original.href = url.href; original.target = "_blank"; original.rel = "noopener noreferrer";
          item.append(original);
        }
      } catch (_) { /* 来源缺少有效在线地址时仍可查看本地文档。 */ }
    }
    details.append(item);
  }
  container.replaceChildren(details);
}

function feedbackIcon(direction) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  svg.classList.add("feedback-icon");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", "M7 10v11H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3Zm0 0 5-8c1.4 0 2.3 1.2 2 2.6L13 10h5.7a2 2 0 0 1 2 2.4l-1.5 7A2 2 0 0 1 17.2 21H7");
  path.setAttribute("fill", "none");
  path.setAttribute("stroke", "currentColor");
  path.setAttribute("stroke-width", "1.7");
  path.setAttribute("stroke-linecap", "round");
  path.setAttribute("stroke-linejoin", "round");
  svg.append(path);
  if (direction === "down") svg.classList.add("feedback-icon-down");
  return svg;
}

function addFeedbackActions(container, feedbackId, alreadyLiked = false) {
  if (isGuest) return;
  if (!feedbackId) return;
  const actions = node("div", "feedback-actions");
  const prompt = node("span", "feedback-prompt", "这个回答有帮助吗？");
  const up = node("button", "feedback-button", "");
  const down = node("button", "feedback-button", "");
  const status = node("span", "feedback-status", "");
  up.type = down.type = "button";
  up.title = "有帮助，加入所有账户共享的问答知识库";
  up.setAttribute("aria-label", up.title);
  down.title = "没有帮助";
  down.setAttribute("aria-label", down.title);
  up.append(feedbackIcon("up"));
  down.append(feedbackIcon("down"));
  up.addEventListener("click", async () => {
    if (up.disabled) return;
    down.classList.remove("selected");
    up.classList.add("selected");
    up.disabled = true;
    status.textContent = "正在保存…";
    try {
      const response = await secureFetch("/api/feedback", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ feedback_id: feedbackId })
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "点赞保存失败，请重试。");
      down.disabled = true;
      status.textContent = result.already_saved ? "已在问答知识库中" : "已加入问答知识库";
    } catch (error) {
      up.disabled = false;
      up.classList.remove("selected");
      status.textContent = error.message || "保存失败，请重新点击 👍。";
    }
  });
  down.addEventListener("click", () => {
    up.classList.remove("selected");
    down.classList.add("selected");
    status.textContent = "已收到反馈";
  });
  actions.append(prompt, up, down, status);
  if (alreadyLiked) {
    up.disabled = down.disabled = true; up.classList.add("selected"); status.textContent = "已加入共享知识库";
  }
  container.replaceChildren(actions);
}

async function submitQuestion(question) {
  question = question.trim();
  if (!question || busy || !activeConversation) return;
  if (currentStatus && !currentStatus.index_ready) { showError("请先点击“更新文档索引”建立文档库。"); return; }
  $("error").classList.add("hidden");
  $("welcome").classList.add("hidden");
  const turnId = `turn-${Date.now()}`;
  const turn = node("article", "turn");
  const answer = node("div", "answer-body markdown pending", "正在检索相关文档…");
  const sources = node("div");
  const notices = node("div", "notices");
  const meta = node("div", "answer-meta");
  const feedback = node("div", "feedback-container");
  const time = node("span");
  const copy = node("button", "copy-button", "复制回答"); copy.type = "button"; copy.disabled = true;
  meta.append(time, copy);
  turn.append(node("div", "question-bubble", question), node("div", "answer-label", "领星文档助手"), answer, sources, notices, meta, feedback);
  $("conversation").append(turn);
  $("question").value = "";
  setBusy(true);
  let rawAnswer = "";
  let done = false;
  const wasNearBottom = () => window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 160;
  function onEvent(event) {
    if (event.type === "status") {
      $("activity").textContent = event.message;
      if (!rawAnswer) answer.textContent = event.message;
    } else if (event.type === "sources") {
      showSources(sources, event.sources, turnId);
    } else if (event.type === "token") {
      const follow = wasNearBottom();
      rawAnswer += event.text;
      answer.className = "answer-body markdown streaming";
      answer.textContent = rawAnswer;
      if (follow) window.scrollTo({ top: document.body.scrollHeight });
    } else if (event.type === "done") {
      done = true;
      rawAnswer = event.answer;
      answer.className = "answer-body markdown";
      // HTML 只使用后端 markdown + bleach 清理后的结果。
      answer.innerHTML = event.answer_html;
      answer.querySelectorAll('a[href^="#source-"]').forEach((link) => {
        const sourceId = link.getAttribute("href").slice(8);
        link.href = `#${turnId}-${sourceId}`;
        link.addEventListener("click", () => {
          const target = document.getElementById(`${turnId}-${sourceId}`);
          if (target) target.closest(".sources").open = true;
        });
      });
      showSources(sources, event.sources, turnId);
      notices.textContent = event.notices.join(" ");
      time.textContent = `${event.stats.total_seconds}s · ${event.sources.length} 处来源`;
      if (event.stats.rerank_applied) time.textContent += ` · 重排序 ${event.stats.rerank_seconds}s`;
      if (event.stats.query_decomposition_applied) time.textContent += ` · 拆分 ${event.stats.query_decomposition_count} 个子问题`;
      if (event.stats.rerank_low_confidence) time.textContent += " · 相关性不足";
      copy.disabled = false;
      $("memory-label").textContent = event.memory_version ? `记忆摘要 · 第 ${event.memory_version} 版` : "近期问答自动记忆";
      loadConversationList().catch(problem => showError(problem.message));
      if (event.mode === "answer") {
        addFeedbackActions(feedback, event.feedback_id);
      }
    } else if (event.type === "error") { throw new Error(event.message); }
  }
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(rawAnswer); copy.textContent = "已复制"; setTimeout(() => { copy.textContent = "复制回答"; }, 1500); }
    catch (_) { showError("复制失败，请选中回答后手动复制。"); }
  });
  try {
    const response = await secureFetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: activeConversation, request_id: crypto.randomUUID(), category: $("category").value, mode: $("retrieve-only").checked ? "retrieve" : "answer" })
    });
    if (!response.ok) { const error = await response.json(); throw new Error(error.error || "请求失败"); }
    if (!response.body) throw new Error("浏览器不支持流式响应，请更换新版浏览器。");
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const result = await reader.read();
        buffer += decoder.decode(result.value || new Uint8Array(), { stream: !result.done });
        let newline;
        while ((newline = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, newline).trim(); buffer = buffer.slice(newline + 1);
          if (line) onEvent(JSON.parse(line));
        }
        if (result.done) break;
      }
      if (buffer.trim()) onEvent(JSON.parse(buffer));
      if (!done) throw new Error("连接提前结束，回答可能不完整，请重试。");
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  } catch (error) {
    showError(error.message || "服务连接失败，请重试。");
    if (!rawAnswer) { answer.className = "answer-body"; answer.textContent = "本次未能完成回答，可查看已检索到的来源后重试。"; }
    else time.textContent = "回答未完成";
  } finally {
    setBusy(false); $("activity").textContent = "Enter 发送 · Shift + Enter 换行"; $("question").focus();
  }
}

$("question-form").addEventListener("submit", (event) => { event.preventDefault(); submitQuestion($("question").value); });
$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); submitQuestion($("question").value); }
});
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => submitQuestion(button.dataset.question)));
$("clear-chat").addEventListener("click", newConversation);
$("new-conversation").addEventListener("click", newConversation);
$("older-turns").addEventListener("click", loadOlderTurns);
$("logout").addEventListener("click", async () => {
  if (busy) return;
  try { await jsonRequest("/api/auth/logout", {method: "POST"}); localStorage.removeItem(conversationKey); location.assign("/login"); }
  catch (problem) { showError(problem.message); }
});
$("update-index").addEventListener("click", async () => {
  try {
    const response = await secureFetch("/api/index", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    const data = await response.json(); if (!response.ok) throw new Error(data.error); await loadStatus();
  } catch (error) { showError(error.message); }
});

async function loadConversationList() {
  const data = await jsonRequest("/api/conversations");
  const list = $("conversation-list");
  list.replaceChildren();
  for (const conversation of data.conversations) {
    const button = node("button", "conversation-choice", conversation.title);
    button.type = "button"; button.title = conversation.title; button.disabled = busy;
    if (conversation.id === activeConversation) {
      button.classList.add("active");
      $("conversation-title").textContent = conversation.title;
      $("memory-label").textContent = conversation.memory_version ? `记忆摘要 · 第 ${conversation.memory_version} 版` : "近期问答自动记忆";
    }
    button.addEventListener("click", () => openConversation(conversation.id));
    list.append(button);
  }
  return data.conversations;
}

function savedTurn(row) {
  const event = row.response;
  const turnId = `saved-turn-${row.id}`;
  const turn = node("article", "turn");
  const answer = node("div", "answer-body markdown");
  answer.innerHTML = event.answer_html;
  answer.querySelectorAll('a[href^="#source-"]').forEach(link => {
    const sourceId = link.getAttribute("href").slice(8);
    link.href = `#${turnId}-${sourceId}`;
    link.addEventListener("click", () => {
      const target = document.getElementById(`${turnId}-${sourceId}`);
      if (target) target.closest(".sources").open = true;
    });
  });
  const sources = node("div"); showSources(sources, event.sources || [], turnId);
  const feedback = node("div", "feedback-container");
  if (event.mode === "answer") addFeedbackActions(feedback, row.feedback_id, row.liked);
  const copy = node("button", "copy-button", "复制回答");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(row.answer); copy.textContent = "已复制"; }
    catch (_) { showError("复制失败，请手动选择回答复制。"); }
  });
  turn.append(node("div", "question-bubble", row.question), node("div", "answer-label", "领星文档助手"), answer,
              sources, node("div", "notices", (event.notices || []).join(" ")), copy, feedback);
  return turn;
}

async function openConversation(id) {
  if (busy) return;
  setBusy(true);
  try {
    const data = await jsonRequest(`/api/conversations/${encodeURIComponent(id)}`);
    activeConversation = id; localStorage.setItem(conversationKey, id);
    olderBefore = data.before;
    $("conversation").replaceChildren(...data.turns.map(savedTurn));
    $("welcome").classList.toggle("hidden", data.turns.length > 0);
    $("older-turns").classList.toggle("hidden", !data.has_more);
    $("error").classList.add("hidden");
    if (data.memory_error) showError(data.memory_error);
    $("question").value = "";
    await loadConversationList();
  } catch (problem) { showError(problem.message); }
  finally { setBusy(false); }
}

async function newConversation() {
  if (busy) return;
  setBusy(true);
  try {
    const data = await jsonRequest("/api/conversations", {method: "POST"});
    setBusy(false);
    await openConversation(data.id);
    $("question").focus();
  } catch (problem) { showError(problem.message); setBusy(false); }
}

async function loadOlderTurns() {
  if (busy || !activeConversation || !olderBefore) return;
  setBusy(true);
  try {
    const data = await jsonRequest(`/api/conversations/${activeConversation}?before=${olderBefore}`);
    const previousHeight = document.documentElement.scrollHeight;
    $("conversation").prepend(...data.turns.map(savedTurn));
    olderBefore = data.before;
    $("older-turns").classList.toggle("hidden", !data.has_more);
    window.scrollBy(0, document.documentElement.scrollHeight - previousHeight);
  } catch (problem) { showError(problem.message); }
  finally { setBusy(false); }
}

async function initializeConversations() {
  setBusy(true);
  try {
    const conversations = await loadConversationList();
    const previous = localStorage.getItem(conversationKey);
    const id = conversations.find(item => item.id === previous)?.id || conversations[0]?.id;
    setBusy(false);
    if (id) await openConversation(id); else await newConversation();
  } catch (problem) { showError(problem.message); setBusy(false); }
}
loadStatus();
initializeConversations();
