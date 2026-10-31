"use strict";
const form = document.getElementById("auth-form");
document.getElementById("guest-login").addEventListener("click", async () => {
  const button = document.getElementById("guest-login");
  const error = document.getElementById("auth-error");
  if (button.disabled || document.getElementById("auth-submit").disabled) return;
  button.disabled = true; error.classList.add("hidden");
  document.getElementById("auth-submit").disabled = true;
  try {
    const response = await fetch("/api/auth/guest", {
      method: "POST", headers: {"X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content}
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "游客登录失败，请重试。");
    location.assign(data.redirect);
  } catch (problem) {
    error.textContent = problem.message || "无法连接服务。";
    error.classList.remove("hidden"); button.disabled = false;
    document.getElementById("auth-submit").disabled = false;
  }
});
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.getElementById("auth-submit");
  const error = document.getElementById("auth-error");
  if (button.disabled || document.getElementById("guest-login").disabled) return;
  const previousLabel = button.textContent;
  button.disabled = true; button.textContent = "请稍候…"; error.classList.add("hidden");
  document.getElementById("guest-login").disabled = true;
  try {
    const response = await fetch(`/api/auth/${form.dataset.mode}`, {
      method: "POST", headers: {"Content-Type": "application/json", "X-CSRF-Token": document.querySelector('meta[name="csrf-token"]').content},
      body: JSON.stringify(Object.fromEntries(new FormData(form)))
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "操作失败，请重试。");
    location.assign(data.redirect);
  } catch (problem) {
    error.textContent = problem.message || "无法连接服务。"; error.classList.remove("hidden");
    button.disabled = false; button.textContent = previousLabel;
    document.getElementById("guest-login").disabled = false;
  }
});
