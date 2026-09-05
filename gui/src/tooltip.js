// Reusable info-tooltip: any <button class="info-tip" data-tooltip="...">
// shows its tooltip text on hover or keyboard focus. No per-instance JS needed
// beyond calling initTooltips() once.
export function initTooltips(root = document) {
  const triggers = root.querySelectorAll(".info-tip:not([data-tooltip-ready])");

  for (const trigger of triggers) {
    trigger.dataset.tooltipReady = "true";
    trigger.setAttribute("type", "button");
    trigger.setAttribute("aria-label", "More info");

    const bubble = document.createElement("span");
    bubble.className = "info-tip-bubble";
    bubble.textContent = trigger.dataset.tooltip || "";
    bubble.setAttribute("role", "tooltip");
    trigger.appendChild(bubble);

    // Touch devices have no hover; let a tap toggle the bubble instead.
    trigger.addEventListener("click", (event) => {
      event.preventDefault();
      trigger.classList.toggle("info-tip-open");
    });

    trigger.addEventListener("blur", () => {
      trigger.classList.remove("info-tip-open");
    });
  }
}
