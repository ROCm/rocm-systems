// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// A native `title` attribute gets dismissed by a kernel row's click-to-filter
// handler and won't reappear until the pointer leaves and re-enters, so the
// kernel list uses this mouse-following tooltip to show the full kernel name
// instead.
(function () {
  "use strict";

  var NAME_TOOLTIP_OFFSET = 14;
  var NAME_TOOLTIP_DELAY_MS = 2000;

  var nameTooltip = null;
  var nameTooltipTimer = null;

  function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), maximum);
  }

  function ensureNameTooltip() {
    if (!nameTooltip) {
      nameTooltip = document.createElement("div");
      nameTooltip.className = "roofline-name-tooltip";
      nameTooltip.setAttribute("role", "tooltip");
      document.body.appendChild(nameTooltip);
    }
    return nameTooltip;
  }

  function positionNameTooltip(x, y) {
    var tooltip = nameTooltip;
    if (!tooltip) {
      return;
    }
    var maxLeft = Math.max(4, window.innerWidth - tooltip.offsetWidth - 4);
    var maxTop = Math.max(4, window.innerHeight - tooltip.offsetHeight - 4);
    tooltip.style.left = clamp(x + NAME_TOOLTIP_OFFSET, 4, maxLeft) + "px";
    tooltip.style.top = clamp(y + NAME_TOOLTIP_OFFSET, 4, maxTop) + "px";
  }

  function showNameTooltip(text, x, y) {
    var tooltip = ensureNameTooltip();
    tooltip.textContent = text;
    tooltip.classList.add("visible");
    positionNameTooltip(x, y);
  }

  function scheduleNameTooltip(text, getPosition) {
    clearTimeout(nameTooltipTimer);
    nameTooltipTimer = setTimeout(function () {
      var position = getPosition();
      showNameTooltip(text, position.x, position.y);
    }, NAME_TOOLTIP_DELAY_MS);
  }

  function hideNameTooltip() {
    clearTimeout(nameTooltipTimer);
    if (nameTooltip) {
      nameTooltip.classList.remove("visible");
    }
  }

  function attach(label, action, title) {
    var lastMousePosition = { x: 0, y: 0 };
    label.addEventListener("mouseenter", function (event) {
      lastMousePosition = { x: event.clientX, y: event.clientY };
      scheduleNameTooltip(title, function () {
        return lastMousePosition;
      });
    });
    label.addEventListener("mousemove", function (event) {
      lastMousePosition = { x: event.clientX, y: event.clientY };
      if (nameTooltip && nameTooltip.classList.contains("visible")) {
        positionNameTooltip(event.clientX, event.clientY);
      }
    });
    label.addEventListener("mouseleave", hideNameTooltip);
    action.addEventListener("focus", function () {
      var rect = label.getBoundingClientRect();
      scheduleNameTooltip(title, function () {
        return { x: rect.left, y: rect.bottom };
      });
    });
    action.addEventListener("blur", hideNameTooltip);
  }

  window.RooflineKernelListTooltip = {
    attach: attach,
    hide: hideNameTooltip,
  };
})();
