/* Install-page selector.
 *
 * Content ships visible; this script hides what does not match the current
 * selection. If it fails to run the page still shows every method.
 */
(function () {
  "use strict";

  var STORAGE_KEY = "amdsmi-install-selector";

  var root = document.querySelector("[data-amdsmi-selector]");
  if (!root) {
    return;
  }

  var config;
  try {
    config = JSON.parse(root.getAttribute("data-amdsmi-selector"));
  } catch (e) {
    return;
  }

  var axes = config.axes || {};
  // axis -> method values that make the axis relevant; absent means always.
  var appliesTo = config.appliesTo || {};
  var axisNames = Object.keys(axes);

  function readSelection(el) {
    try {
      return JSON.parse(el.getAttribute("data-selector")) || {};
    } catch (e) {
      return {};
    }
  }

  var sections = [];
  Array.prototype.forEach.call(
    document.querySelectorAll(".amdsmi-section-marker"),
    function (marker) {
      var section = marker.closest("section");
      if (section) {
        sections.push({ el: section, selection: readSelection(marker) });
      }
    }
  );

  function selectionFromClasses(el) {
    var selection = {};
    Array.prototype.forEach.call(el.classList, function (name) {
      var match = /^amdsmi-when--([a-z]+)-(.+)$/.exec(name);
      if (!match) {
        return;
      }
      selection[match[1]] = (selection[match[1]] || []).concat(match[2]);
    });
    return selection;
  }

  var blocks = Array.prototype.map.call(
    document.querySelectorAll(".amdsmi-when"),
    function (el) {
      return { el: el, selection: selectionFromClasses(el) };
    }
  );

  function matches(selection, state) {
    for (var axis in selection) {
      if (!Object.prototype.hasOwnProperty.call(selection, axis)) {
        continue;
      }
      var accepted = selection[axis];
      if (!accepted || !accepted.length) {
        continue;
      }
      if (accepted.indexOf(state[axis]) === -1) {
        return false;
      }
    }
    return true;
  }

  function axisApplies(axis, state) {
    var limit = appliesTo[axis];
    return !limit || limit.indexOf(state.method) !== -1;
  }

  var state = {};
  axisNames.forEach(function (axis) {
    state[axis] = axes[axis][0];
  });

  try {
    var saved = JSON.parse(window.localStorage.getItem(STORAGE_KEY)) || {};
    axisNames.forEach(function (axis) {
      if (axes[axis].indexOf(saved[axis]) !== -1) {
        state[axis] = saved[axis];
      }
    });
  } catch (e) {
    /* private mode or malformed value: keep defaults */
  }

  var params = new URLSearchParams(window.location.search);
  axisNames.forEach(function (axis) {
    var value = params.get(axis);
    if (value && axes[axis].indexOf(value) !== -1) {
      state[axis] = value;
    }
  });

  /* Reveal whatever the URL fragment points at, so cross-page links into a
   * non-selected method still land on visible content. */
  function rescueAnchor() {
    var hash = window.location.hash;
    if (!hash || hash.length < 2) {
      return;
    }
    var target;
    try {
      target = document.querySelector(hash);
    } catch (e) {
      return;
    }
    if (!target) {
      return;
    }
    var node = target;
    while (node && node !== document.body) {
      var required = null;
      if (node.classList && node.classList.contains("amdsmi-when")) {
        required = selectionFromClasses(node);
      } else if (node.tagName === "SECTION") {
        var marker = node.querySelector(":scope > .amdsmi-section-marker");
        if (marker) {
          required = readSelection(marker);
        }
      }
      if (required) {
        axisNames.forEach(function (axis) {
          var accepted = required[axis];
          if (accepted && accepted.length && accepted.indexOf(state[axis]) === -1) {
            state[axis] = accepted[0];
          }
        });
      }
      node = node.parentNode;
    }
  }

  function syncToc() {
    var links = document.querySelectorAll(".bd-toc a, .toc-entry a");
    Array.prototype.forEach.call(links, function (link) {
      var href = link.getAttribute("href") || "";
      if (href.charAt(0) !== "#") {
        return;
      }
      var item = link.closest("li");
      var target = document.getElementById(href.slice(1));
      if (!item || !target) {
        return;
      }
      item.hidden = target.closest("[hidden]") !== null;
    });
  }

  function apply() {
    axisNames.forEach(function (axis) {
      var group = root.querySelector(
        '.amdsmi-selector__axis[data-axis="' + axis + '"]'
      );
      if (group) {
        group.hidden = !axisApplies(axis, state);
      }
      var options = root.querySelectorAll(
        '.amdsmi-selector__axis[data-axis="' + axis + '"] .amdsmi-selector__option'
      );
      Array.prototype.forEach.call(options, function (button) {
        button.setAttribute(
          "aria-pressed",
          String(button.getAttribute("data-value") === state[axis])
        );
      });
    });

    sections.forEach(function (entry) {
      entry.el.hidden = !matches(entry.selection, state);
    });
    blocks.forEach(function (entry) {
      entry.el.hidden = !matches(entry.selection, state);
    });

    syncToc();
  }

  function save() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      /* storage unavailable: selection just does not persist */
    }
  }

  function syncUrl() {
    if (!window.history || !window.history.replaceState) {
      return;
    }
    var next = new URLSearchParams(window.location.search);
    axisNames.forEach(function (axis) {
      if (axisApplies(axis, state)) {
        next.set(axis, state[axis]);
      } else {
        next.delete(axis);
      }
    });
    var query = next.toString();
    window.history.replaceState(
      null,
      "",
      window.location.pathname +
        (query ? "?" + query : "") +
        window.location.hash
    );
  }

  root.addEventListener("click", function (event) {
    var button = event.target.closest(".amdsmi-selector__option");
    if (!button || !root.contains(button)) {
      return;
    }
    var group = button.closest(".amdsmi-selector__axis");
    if (!group) {
      return;
    }
    var axis = group.getAttribute("data-axis");
    var value = button.getAttribute("data-value");
    if (!axes[axis] || axes[axis].indexOf(value) === -1) {
      return;
    }
    state[axis] = value;
    apply();
    save();
    syncUrl();
  });

  window.addEventListener("hashchange", function () {
    rescueAnchor();
    apply();
    syncUrl();
  });

  rescueAnchor();
  apply();
  syncUrl();

  if (window.location.hash) {
    try {
      var landing = document.querySelector(window.location.hash);
      if (landing) {
        landing.scrollIntoView();
      }
    } catch (e) {
      /* malformed fragment */
    }
  }
})();
