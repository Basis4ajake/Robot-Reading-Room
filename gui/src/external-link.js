// Links to the outside web (e.g. the Ollama model catalog) must open in the
// system browser, not navigate the Tauri webview itself. When running inside
// Tauri, hand the click to the opener plugin; outside Tauri (e.g. `tauri dev`
// loaded in a plain browser tab) fall back to the anchor's normal
// target="_blank" navigation.
export function initExternalLinks(root = document) {
  const links = root.querySelectorAll("a[data-external-link]:not([data-external-ready])");

  for (const link of links) {
    link.dataset.externalReady = "true";
    link.addEventListener("click", (event) => {
      const opener = window.__TAURI__?.opener;
      if (opener?.openUrl) {
        event.preventDefault();
        opener.openUrl(link.href);
      }
    });
  }
}
