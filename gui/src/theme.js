const STORAGE_KEY = "rrr-theme";

let toggleBtn;

function applyTheme(theme) {
  if (theme === "bendy") {
    document.documentElement.setAttribute("data-theme", "bendy");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
}

function readStoredTheme() {
  try {
    return localStorage.getItem(STORAGE_KEY) || "plain";
  } catch {
    return "plain";
  }
}

function writeStoredTheme(theme) {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // localStorage unavailable - theme just won't persist across restarts.
  }
}

function updateToggleLabel(theme) {
  toggleBtn.textContent = theme === "bendy" ? "Switch to Plain" : "Switch to Bendy Toon";
}

export function initTheme() {
  toggleBtn = document.querySelector("#theme-toggle-btn");

  const theme = readStoredTheme();
  applyTheme(theme);
  updateToggleLabel(theme);

  toggleBtn.addEventListener("click", () => {
    const next = readStoredTheme() === "bendy" ? "plain" : "bendy";
    writeStoredTheme(next);
    applyTheme(next);
    updateToggleLabel(next);
  });
}
