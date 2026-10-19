// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
(function () {
  "use strict";

  var DARK_THEME_CLASS = "report-theme-dark";

  function readModel(modelId) {
    var modelEl = document.getElementById(modelId);
    if (!modelEl) {
      return null;
    }
    try {
      return JSON.parse(modelEl.textContent);
    } catch (err) {
      return null;
    }
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
  };
})();
