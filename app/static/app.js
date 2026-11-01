"use strict";
const $ = (id) => document.getElementById(id);
let busy = false;
let activeConversation = null;
let olderBefore = null;
let currentGeneration = null;
let conversationListSequence = 0;
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
  $("conversation-search").disabled = value;
  $("older-turns").disabled = value;
  $("logout").disabled = value;
  for (const id of ["rename-conversation", "export-conversation", "delete-conversation"]) $(id).disabled = value || !activeConversation;
  $("stop-generation").classList.toggle("hidden", !value || !currentGeneration);
  $("stop-generation").disabled = !currentGeneration || !!currentGeneration.stopping;
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
    const sync = status.document_sync || {};
    $("update-index").disabled = isGuest || status.job.running || sync.running;
    $("check-docs").disabled = isGuest || !sync.enabled || sync.running;
    $("sync-docs").disabled = isGuest || !sync.enabled || sync.running || status.job.running;
    const countChanges = changes => Object.values(changes || {}).reduce((sum, paths) => sum + (Array.isArray(paths) ? paths.length : 0), 0);
    $("sync-status").textContent = sync.running ? sync.message : sync.error || sync.last_error ||
      (!sync.enabled ? "自动同步已关闭" : `每 ${Math.round((sync.interval_seconds || 3600) / 60)} 分钟检查 · 最近检查：${formatDate(sync.checked_at)}`);
    const available = sync.available_changes_counts ? Object.values(sync.available_changes_counts).reduce((a, b) => a + b, 0) : countChanges(sync.available_changes);
    if (!sync.running && available) $("sync-status").textContent += ` · 发现 ${available} 处变化待同步`;
    if (!status.llm_configured) {
      $("config-notice").textContent = "在 config_local.py 中配置 DEEPSEEK_API_KEY 并重启后，即可生成回答。当前可使用检索预览。";
      $("config-notice").classList.remove("hidden");
    } else { $("config-notice").classList.add("hidden"); }
    const selected = $("category").value;
    const optionValues = [...$("category").options].map((option) => option.value).slice(1);
    if (JSON.stringify(optionValues) !== JSON.stringify(status.categories)) {
      $("category").replaceChildren(new Option("全部业务", ""), ...status.categories.map((name) => new Option(name, name)));
      if (status.categories.includes(selected)) $("category").value = selected;
    }
    if (status.job.error) showError(status.job.error);
    if (!statusTimer) statusTimer = setInterval(loadStatus, 5000);
    if ($("sync-history").open) await loadDocumentHistory();
  } catch (error) { $("index-label").textContent = "服务连接失败"; showError(error.message); }
}

function showSources(container, sources, turnId) {
  const details = node("details", "sources");
  const dates = [...new Set(sources.map(source => source.updated_at?.slice(0, 10)).filter(Boolean))].sort();
  const freshness = dates.length ? ` · 资料版本 ${dates[0]}${dates.length > 1 ? `～${dates.at(-1)}` : ""}` : "";
  details.append(node("summary", "", `依据与参考 · ${sources.length} 处来源${freshness}`));
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
    if (source.updated_at) {
      const text = `资料版本：${formatDate(source.updated_at)} · 同步：${formatDate(source.synced_at)}`;
      const version = node("p", "source-freshness", text);
      version.title = `上游提交 ${source.source_revision || "未知"}。这是所同步来源版本的时间，不等同于官方发布日期。`;
      item.append(version);
    }
    if (source.source_type === "liked_qa") item.append(node("span", "source-badge", "点赞问答 · 辅助参考"));
    if (source.source_type === "document_version") item.append(node("span", "source-badge", "本机版本记录"));
    item.append(node("p", "", [source.category, source.heading].filter(Boolean).join(" / ")));
    if (source.api_path) item.append(node("p", "", `${source.method} ${source.api_path}`));
    const excerpt = node("details", "source-excerpt");
    excerpt.append(node("summary", "", "检索片段"), node("pre", "", source.excerpt));
    item.append(excerpt);
    if (source.source_url) {
      try {
        const url = new URL(source.source_url);
        if (["https:", "http:"].includes(url.protocol)) {
          const original = node("a", "", source.source_type === "document_version" ? "文档来源仓库 ↗" : "官方原文 ↗");
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
  const generation = {requestId: crypto.randomUUID(), conversationId: activeConversation, controller: new AbortController(), stopped: false, stopping: false, completed: false};
  currentGeneration = generation;
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
      generation.completed = true;
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
      if (event.stats.retrieval_intent) {
        const labels = {endpoint: "接口", parameters: "参数", response: "返回字段", workflow: "流程", version: "版本", general: "文档"};
        time.textContent += ` · ${labels[event.stats.retrieval_intent] || "文档"}检索 ${event.stats.retrieval_rounds} 轮`;
      }
      if (event.stats.needs_clarification) time.textContent += " · 待确认接口";
      if (event.stats.rerank_low_confidence) time.textContent += " · 相关性不足";
      copy.disabled = false;
      $("memory-label").textContent = event.memory_version ? `记忆摘要 · 第 ${event.memory_version} 版` : "近期问答自动记忆";
      loadConversationList().catch(problem => showError(problem.message));
      if (event.mode === "answer") {
        addFeedbackActions(feedback, event.feedback_id);
      }
    } else if (event.type === "cancelled") {
      generation.stopped = true;
      generation.controller.abort();
    } else if (event.type === "error") { throw new Error(event.message); }
  }
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(rawAnswer); copy.textContent = "已复制"; setTimeout(() => { copy.textContent = "复制回答"; }, 1500); }
    catch (_) { showError("复制失败，请选中回答后手动复制。"); }
  });
  try {
    const response = await secureFetch("/api/ask/stream", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: generation.controller.signal,
      body: JSON.stringify({ question, conversation_id: generation.conversationId, request_id: generation.requestId, category: $("category").value, mode: $("retrieve-only").checked ? "retrieve" : "answer" })
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
      if (!done && !generation.stopped) throw new Error("连接提前结束，回答可能不完整，请重试。");
    } finally { await reader.cancel().catch(() => {}); reader.releaseLock(); }
  } catch (error) {
    if (generation.stopped) {
      answer.className = "answer-body markdown";
      answer.textContent = rawAnswer || "已停止生成。";
      time.textContent = "已停止 · 保存前取消的回答不计入历史和记忆";
      copy.disabled = !rawAnswer;
    } else {
    showError(error.message || "服务连接失败，请重试。");
    if (!rawAnswer) { answer.className = "answer-body"; answer.textContent = "本次未能完成回答，可查看已检索到的来源后重试。"; }
    else time.textContent = "回答未完成";
    }
  } finally {
    if (generation.stopped && !done) {
      answer.className = "answer-body markdown";
      answer.textContent = rawAnswer || "已停止生成。";
      time.textContent = "已停止 · 本次未完成的回答不保存";
      copy.disabled = !rawAnswer;
    }
    if (currentGeneration === generation) currentGeneration = null;
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
  const sequence = ++conversationListSequence;
  const query = $("conversation-search").value.trim();
  const data = await jsonRequest(`/api/conversations?q=${encodeURIComponent(query)}`);
  if (sequence !== conversationListSequence) return data.conversations;
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
  if (!data.conversations.length) list.append(node("p", "sidebar-note", query ? "没有找到匹配的对话" : "还没有对话"));
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
    $("conversation-title").textContent = data.title;
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
function formatDate(value) {
  if (!value) return "尚无记录";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "时间未知" : date.toLocaleString("zh-CN", {year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false});
}

$("stop-generation").addEventListener("click", async () => {
  const generation = currentGeneration;
  if (!generation || generation.completed || generation.stopping) return;
  generation.stopping = true;
  $("stop-generation").disabled = true;
  $("activity").textContent = "正在停止生成…";
  try {
    const result = await jsonRequest(`/api/conversations/${generation.conversationId}/cancel`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({request_id: generation.requestId})
    });
    if (!generation.completed && !result.already_completed) {
      generation.stopped = true;
      generation.controller.abort();
    }
  } catch (problem) {
    showError(`停止请求失败：${problem.message}`);
    generation.stopping = false;
    if (currentGeneration === generation) $("stop-generation").disabled = false;
  }
});

let searchTimer = null;
$("conversation-search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => loadConversationList().catch(problem => showError(problem.message)), 220);
});

$("rename-conversation").addEventListener("click", async () => {
  if (busy || !activeConversation) return;
  const title = window.prompt("对话名称（1–100 个字符）", $("conversation-title").textContent);
  if (title === null) return;
  setBusy(true);
  try {
    const result = await jsonRequest(`/api/conversations/${activeConversation}`, {
      method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({title})
    });
    $("conversation-title").textContent = result.title;
    await loadConversationList();
  } catch (problem) { showError(problem.message); }
  finally { setBusy(false); }
});

$("delete-conversation").addEventListener("click", async () => {
  if (busy || !activeConversation) return;
  if (!window.confirm("删除当前对话、全部历史问答和摘要？此操作不可恢复。已点赞的共享知识仍会保留。")) return;
  setBusy(true);
  try {
    const result = await jsonRequest(`/api/conversations/${activeConversation}`, {method: "DELETE"});
    activeConversation = null;
    localStorage.removeItem(conversationKey);
    $("conversation-search").value = "";
    const conversations = await loadConversationList();
    const next = result.replacement?.id || conversations[0]?.id;
    setBusy(false);
    if (next) await openConversation(next); else await newConversation();
  } catch (problem) { showError(problem.message); setBusy(false); }
});

$("export-conversation").addEventListener("click", async () => {
  if (busy || !activeConversation) return;
  const conversationId = activeConversation;
  const filename = $("conversation-title").textContent.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").slice(0, 100) || "对话";
  setBusy(true);
  try {
    const response = await secureFetch(`/api/conversations/${conversationId}/export`);
    if (!response.ok) { const error = await response.json(); throw new Error(error.error || "导出失败"); }
    const url = URL.createObjectURL(await response.blob());
    const link = node("a");
    link.href = url; link.download = `${filename}.md`;
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (problem) { showError(problem.message); }
  finally { setBusy(false); }
});

async function requestDocumentSync(checkOnly) {
  $("check-docs").disabled = $("sync-docs").disabled = true;
  try {
    await jsonRequest("/api/documents/sync", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({check_only: checkOnly})});
    await loadStatus();
  } catch (problem) { showError(problem.message); await loadStatus(); }
}
$("check-docs").addEventListener("click", () => requestDocumentSync(true));
$("sync-docs").addEventListener("click", () => requestDocumentSync(false));
$("sync-history").addEventListener("toggle", () => {
  if ($("sync-history").open) loadDocumentHistory().catch(problem => showError(problem.message));
});

async function loadDocumentHistory() {
  const data = await jsonRequest("/api/documents/versions");
  const container = $("sync-versions");
  container.replaceChildren();
  if (!data.versions.length) { container.append(node("p", "sidebar-note", "同步后会在这里记录文档版本与增删改明细。")); return; }
  for (const version of data.versions) {
    const item = node("details", "sync-version");
    item.append(node("summary", "", `${version.revision.slice(0, 8)} · ${formatDate(version.synced_at)}`));
    item.append(node("p", "", `来源版本时间：${formatDate(version.source_time)}（非官方发布日期）`));
    for (const [key, label] of [["added", "新增"], ["modified", "修改"], ["deleted", "删除"]]) {
      const paths = version[key] || [];
      const total = version.counts?.[key] ?? paths.length;
      if (!total) continue;
      item.append(node("p", "", `${label} ${total} 篇：${paths.slice(0, 12).join("、")}${total > 12 ? "…" : ""}`));
    }
    if (!(version.added.length + version.modified.length + version.deleted.length)) item.append(node("p", "", "建立资料版本基线，文档内容未变化。"));
    if (version.compare_url?.startsWith("https://github.com/")) {
      const link = node("a", "", "查看上游提交 ↗");
      link.href = version.compare_url; link.target = "_blank"; link.rel = "noopener noreferrer";
      item.append(link);
    }
    container.append(item);
  }
}

loadStatus();
initializeConversations();
