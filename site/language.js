"use strict";

// Explicit page URLs always win. Only the root entry chooses a preferred language.
(() => {
  const language = document.documentElement.lang;
  let preferred;
  try {
    preferred = localStorage.getItem("jevany-language");
  } catch { /* Language links also work when storage is unavailable. */ }
  if (!["en", "zh-CN"].includes(preferred)) {
    preferred = navigator.language.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
  }
  if (location.pathname.endsWith("/") && preferred !== language) {
    location.replace(`zh.html${location.search}${location.hash}`);
    return;
  }
  try {
    localStorage.setItem("jevany-language", language);
  } catch { /* The explicit language URL still preserves the selection. */ }
  document.addEventListener("DOMContentLoaded", () => {
    const updateLinks = () => {
      document.querySelectorAll("[data-language]").forEach(link => {
        const page = link.dataset.language === "zh-CN" ? "zh.html" : "index.html";
        link.setAttribute("href", `${page}${location.search}${location.hash}`);
      });
    };
    updateLinks();
    window.addEventListener("hashchange", updateLinks);
  });
})();

function translate(message, values = {}) {
  const translated = window.jevanyMessages?.[message] ?? message;
  return translated.replace(/\{(\w+)\}/g, (match, key) => values[key] ?? match);
}
