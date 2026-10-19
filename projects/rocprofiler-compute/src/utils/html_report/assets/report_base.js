// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
(function () {
  "use strict";

  var DARK_THEME_CLASS = "report-theme-dark";

  function readModel(modelId) {
    var modelEl = null;
    var elements = document.querySelectorAll("[id]");
    for (var i = 0; i < elements.length; i += 1) {
      if (elements[i].id === modelId) {
        modelEl = elements[i];
        break;
      }
    }
    if (!modelEl) {
      return null;
    }
    try {
      return JSON.parse(modelEl.textContent);
    } catch (err) {
      return null;
    }
  }

  function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), maximum);
  }

  // Kernel's list "click-to-filter" action prevents a default tooltip being shown.
  // Therefore custom tooltip is added to overcome this.
  class KernelNameTooltip {
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
        this.#element.className = "report-name-tooltip";
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

  var kernelNameTooltip = new KernelNameTooltip();

  function createKernelList(options) {
    var items = options.items || [];
    var selected = new Set();
    var rows = new Map();
    var mode = options.mode || "multi";

    function selectedKeys() {
      return new Set(selected);
    }

    function applyRowStates(fn) {
      var filtering = selected.size > 0;
      items.forEach(function (item) {
        var row = rows.get(item.key);
        if (!row) {
          return;
        }
        var isSelected = selected.has(item.key);
        row.classList.toggle("selected", isSelected);
        row.classList.toggle("dimmed", filtering && !isSelected);
        if (fn) {
          fn(row, item);
        }
      });
      if (options.showAllButton) {
        options.showAllButton.disabled = !filtering;
      }
    }

    function select(key, event) {
      if (key !== null && !rows.has(key)) {
        return;
      }
      var before = selectedKeys();
      if (key === null) {
        selected.clear();
      } else if (mode === "multi" && event && (event.ctrlKey || event.metaKey)) {
        if (selected.size === 0) {
          items.forEach(function (item) {
            selected.add(item.key);
          });
          selected.delete(key);
        } else if (selected.has(key)) {
          selected.delete(key);
        } else {
          selected.add(key);
        }
      } else if (selected.size === 1 && selected.has(key)) {
        selected.clear();
      } else {
        selected.clear();
        selected.add(key);
      }
      if (before.size === selected.size &&
          Array.from(before).every(function (itemKey) { return selected.has(itemKey); })) {
        return;
      }
      applyRowStates();
      if (options.onChange) {
        options.onChange(selectedKeys());
      }
      if (selected.size === 1) {
        rows.get(Array.from(selected)[0]).scrollIntoView({ block: "nearest" });
      }
    }

    function setCountText(value) {
      if (options.countElement) {
        options.countElement.textContent = value;
      }
    }

    if (options.list) {
      items.forEach(function (item) {
        var row = document.createElement("li");
        row.className = "report-panel-item";
        row.dataset.key = String(item.key);

        var action = document.createElement("button");
        action.type = "button";
        action.className = "report-panel-action";

        var swatch = document.createElement("span");
        swatch.className = "report-swatch";
        swatch.style.backgroundColor = item.color || "#888888";

        var label = document.createElement("span");
        label.className = "report-panel-name";
        label.textContent = item.label;

        action.appendChild(swatch);
        action.appendChild(label);
        if (item.title) {
          action.setAttribute("aria-label", item.title);
          kernelNameTooltip.attachTooltipToKernelRow(action, label, item.title);
        }
        action.addEventListener("click", function (event) {
          kernelNameTooltip.hide();
          select(item.key, event);
        });
        row.appendChild(action);
        if (options.decorateRow) {
          options.decorateRow(row, item);
        }
        options.list.appendChild(row);
        rows.set(item.key, row);
      });
    }
    if (options.showAllButton) {
      options.showAllButton.addEventListener("click", function () {
        select(null);
      });
    }
    applyRowStates();

    return {
      selectedKeys: selectedKeys,
      select: select,
      applyRowStates: applyRowStates,
      setCountText: setCountText,
    };
  }

  function initTheme(toggle, onChange) {
    var themeIsReaderChoice = false;

    function themeIsDark() {
      return document.documentElement.classList.contains(DARK_THEME_CLASS);
    }

    function syncThemeToggle() {
      if (!toggle) {
        return;
      }
      var dark = themeIsDark();
      toggle.textContent = dark ? "Light mode" : "Dark mode";
      toggle.setAttribute("aria-pressed", String(dark));
      toggle.title = dark
        ? "Switch the page and chart to light colors"
        : "Switch the page and chart to dark colors";
    }

    function setTheme(dark) {
      document.documentElement.classList.toggle(DARK_THEME_CLASS, dark);
      syncThemeToggle();
      onChange(dark);
    }

    syncThemeToggle();
    if (toggle) {
      toggle.addEventListener("click", function () {
        themeIsReaderChoice = true;
        setTheme(!themeIsDark());
      });
    }

    if (typeof window.matchMedia === "function") {
      var query = window.matchMedia("(prefers-color-scheme: dark)");
      var onSystemChange = function (event) {
        if (!themeIsReaderChoice) {
          setTheme(event.matches);
        }
      };
      if (typeof query.addEventListener === "function") {
        query.addEventListener("change", onSystemChange);
      } else if (typeof query.addListener === "function") {
        query.addListener(onSystemChange);
      }
    }
  }

  window.HtmlReport = {
    readModel: readModel,
    initTheme: initTheme,
    createKernelList: createKernelList,
  };
})();
