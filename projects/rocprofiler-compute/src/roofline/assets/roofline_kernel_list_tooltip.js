// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Kernel's list "click-to-filter" action prevents a default tooltip being shown.
// Therefore custom tooltip is added to overcome this.
(function () {
  "use strict";

  function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), maximum);
  }

  class RooflineKernelNameTooltip {
    #offset = 14;
    #delayMs = 2000;
    #element = null;
    #timer = null;

    attachTooltipToKernelRow(kernelRow, kernelLabel, kernelFullName) {
      var self = this;
      var lastMousePosition = { x: 0, y: 0 };
      kernelLabel.addEventListener("mouseenter", function (event) {
        lastMousePosition = { x: event.clientX, y: event.clientY };
        self.#showWithDelay(kernelFullName, function () {
          return lastMousePosition;
        });
      });
      kernelLabel.addEventListener("mousemove", function (event) {
        lastMousePosition = { x: event.clientX, y: event.clientY };
        if (self.#element && self.#element.classList.contains("visible")) {
          self.#position(event.clientX, event.clientY);
        }
      });
      kernelLabel.addEventListener("mouseleave", function () {
        self.hide();
      });
      kernelRow.addEventListener("focus", function () {
        var rect = kernelLabel.getBoundingClientRect();
        self.#showWithDelay(kernelFullName, function () {
          return { x: rect.left, y: rect.bottom };
        });
      });
      kernelRow.addEventListener("blur", function () {
        self.hide();
      });
    }

    hide() {
      clearTimeout(this.#timer);
      if (this.#element) {
        this.#element.classList.remove("visible");
      }
    }

    #showWithDelay(text, getPosition) {
      var self = this;
      clearTimeout(this.#timer);
      this.#timer = setTimeout(function () {
        var position = getPosition();
        self.#show(text, position.x, position.y);
      }, this.#delayMs);
    }

    #show(text, x, y) {
      var tooltip = this.#createElement();
      tooltip.textContent = text;
      tooltip.classList.add("visible");
      this.#position(x, y);
    }

    #createElement() {
      if (!this.#element) {
        this.#element = document.createElement("div");
        this.#element.className = "roofline-name-tooltip";
        this.#element.setAttribute("role", "tooltip");
        document.body.appendChild(this.#element);
      }
      return this.#element;
    }

    #position(x, y) {
      var tooltip = this.#element;
      if (!tooltip) {
        return;
      }
      var maxLeft = Math.max(4, window.innerWidth - tooltip.offsetWidth - 4);
      var maxTop = Math.max(4, window.innerHeight - tooltip.offsetHeight - 4);
      tooltip.style.left = clamp(x + this.#offset, 4, maxLeft) + "px";
      tooltip.style.top = clamp(y + this.#offset, 4, maxTop) + "px";
    }
  }

  var tooltip = new RooflineKernelNameTooltip();

  window.RooflineKernelListTooltip = {
    attachTooltipToKernelRow: tooltip.attachTooltipToKernelRow.bind(tooltip),
    hide: tooltip.hide.bind(tooltip),
  };
})();
