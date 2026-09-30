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
        if (item.pct != null) {
          var pct = document.createElement("span");
          pct.className = "report-kernel-pct";
          pct.textContent = item.pct.toFixed(1) + "%";
          if (options.pctTitle) {
            pct.title = options.pctTitle;
          }
          action.appendChild(pct);
        }
        action.addEventListener("click", function (event) {
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
