// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Client-side controller for the interactive roofline.

(function () {
  "use strict";

  var modelEl = document.getElementById("roofline-model");
  if (!modelEl) {
    return;
  }

  var model;
  try {
    model = JSON.parse(modelEl.textContent);
  } catch (err) {
    return;
  }

  // ---- Config forwarded from roofline_html.py via the model ---------------
  var ALL_PEAKS_VALUE = model.allPeaksValue;
  var ROOF_EXTREME_MAX_AI = model.roofExtremeMaxAi;
  var KERNEL_NAME_FONT_FAMILY = model.kernelNameFontFamily;
  var FRAME_PAD = model.framePad;
  var FRAME_MIN_DECADES = model.frameMinDecades;
  var FRAME_SLOPE_SKEW = model.frameSlopeSkew;

  // ---- Own presentation ---------------------------------------------------
  var ALL_PEAKS_LABEL = "All peaks";
  var FALLBACK_COLOR = "#888888";
  var PLOT_DIM_OPACITY = 0.15;
  var RUNTIME_EPSILON = 1e-6;
  var EXPORT_MIN_WIDTH = 960;
  var EXPORT_MIN_HEIGHT = 560;
  var EXPORT_LEGEND_MIN_WIDTH = 300;
  var EXPORT_LEGEND_MAX_WIDTH = 460;
  var EXPORT_LEGEND_WIDTH_RATIO = 0.34;
  var EXPORT_LEGEND_MAX_HEIGHT_RATIO = 2;
  var EXPORT_LEGEND_MAX_LABEL_LINES = 4;
  var EXPORT_LEGEND_TEXT_INSET = 72;
  var EXPORT_LEGEND_HEADER_HEIGHT = 48;
  var EXPORT_LEGEND_ROW_HEIGHT = 18;
  var EXPORT_LEGEND_FONT_SIZE = 11;
  var EXPORT_LEGEND_FONT_FAMILY = "Arial, sans-serif";
  var EXPORT_ROOF_LEGEND_RANK = 10000;
  var EXPORT_KERNEL_LEGEND_GROUP = "export-kernels";
  var EXPORT_SUBTITLE_HEIGHT = 20;
  var EXPORT_MAX_RASTER_DIMENSION = 30000;
  var EXPORT_MAX_RASTER_AREA = 16.7e6;
  var EXPORT_MAX_SCALE = 4;
  // Poll for Plotly to finish its initial paint before wiring interactivity.
  var PLOT_READY_POLL_MS = 50;
  var PLOT_READY_MAX_ATTEMPTS = 40;

  // ---- DOM handles --------------------------------------------------------
  var gd = document.getElementById(model.divId);
  var peakSelect = document.getElementById("roofline-peak-select");
  var peakControl = document.getElementById("roofline-peak-control");
  var peakControlTitle = peakControl ? peakControl.title : "";
  var kernelList = document.getElementById("roofline-kernel-list");
  var showAllBtn = document.getElementById("roofline-show-all");
  var kernelCountEl = document.getElementById("roofline-kernel-count");
  var runtimeSlider = document.getElementById("roofline-runtime-threshold");
  var runtimeValueEl = document.getElementById("roofline-runtime-value");
  var runtimeFilterEl = document.getElementById("roofline-runtime-filter");
  var roofList = document.getElementById("roofline-roof-list");
  var roofCountEl = document.getElementById("roofline-roof-count");
  var showAllRoofsBtn = document.getElementById("roofline-show-all-roofs");
  var resetViewBtn = document.getElementById("roofline-reset-view");
  var exportPngBtn = document.getElementById("roofline-export-png");
  var themeToggleBtn = document.getElementById("roofline-theme-toggle");
  var plotColumn = gd ? gd.closest(".roofline-plot-col") : null;
  var plotResizeFrame = null;
  var renderFrame = null;
  var exportTextMeasureContext = null;
  var autoFramed = false;
  var applyingFrame = false;
  var themeIsReaderChoice = false;

  // ---- Model data ---------------------------------------------------------
  var kernels = model.kernels;
  // Each kernel's array position is its stable identity for selection,
  // scrolling, and runtime lookup. Names are not unique: two dispatches of the
  // same kernel signature would collapse into one row.
  kernels.forEach(function (kernel, index) {
    kernel.index = index;
  });
  var kernelTraceIndices = model.kernelTraceIndices;
  var rooflineTraces = model.rooflineTraces;
  var computeTraces = model.computeTraces;
  var computeOverlayTraces = model.computeOverlayTraces;
  var peakColors = model.peakColors;
  var initialRange = null;

  function kernelHasRuntime(kernel) {
    return kernel.pctRuntime != null && isFinite(kernel.pctRuntime);
  }

  // Whether any kernel carries a percent-of-runtime, which gates the filter.
  var hasRuntimeData = kernels.some(kernelHasRuntime);

  // Data-driven runtime filter: each kernel's cumulative percent of runtime
  // and the sorted set of those values used as the slider's stops.
  // Filled by computeRuntimeBreakpoints().
  var kernelCumulativePct = {};
  var runtimeBreakpoints = [];

  var memoryRoofIndices = rooflineTraces.map(function (roof) {
    return roof.traceIndex;
  });
  var computeCeilingIndices = computeTraces.map(function (ceiling) {
    return ceiling.traceIndex;
  });

  var state = {
    // The memory region shown in the aggregate view. A single
    // isolated kernel ignores this and shows every level.
    peak: model.defaultPeak || ALL_PEAKS_VALUE,
    // Indices (kernel.index) of the currently isolated kernels.
    selected: new Set(),
    // Trace indices of the memory roofs currently isolated in the legend.
    isolatedRoofs: new Set(),
    // Cumulative-runtime-percent cutoff; a kernel shows when its cumulative
    // percent is within this. Infinity shows every kernel until init sets it.
    runtimeThreshold: Infinity,
  };

  // ===== Small shared helpers ==============================================

  function isMultiSelectEvent(event) {
    return !!(event && (event.ctrlKey || event.metaKey));
  }

  function plotlyReady() {
    return gd && typeof Plotly !== "undefined";
  }

  function toggleSelection(set, key, multi) {
    if (multi) {
      if (set.has(key)) {
        set.delete(key);
      } else {
        set.add(key);
      }
      return;
    }
    if (set.size === 1 && set.has(key)) {
      set.clear();
    } else {
      set.clear();
      set.add(key);
    }
  }

  function kernelIndicesByRuntime() {
    var order = kernels.map(function (_, index) {
      return index;
    });
    order.sort(function (a, b) {
      return (kernels[b].pctRuntime || 0) - (kernels[a].pctRuntime || 0);
    });
    return order;
  }

  function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), maximum);
  }

  function setRowState(item, selected, dimmed) {
    item.classList.toggle("selected", selected);
    item.classList.toggle("dimmed", dimmed);
  }

  // Shared count shown next to each panel title.
  function formatCount(shown, total) {
    return "(" + shown + " / " + total + ")";
  }

  function eachKernelRow(fn) {
    if (!kernelList) {
      return;
    }
    Array.prototype.forEach.call(kernelList.children, function (item) {
      var kernel = kernels[Number(item.dataset.index)];
      if (kernel) {
        fn(item, kernel);
      }
    });
  }

  // ===== The one isolated kernel rule ======================================

  // A single isolated kernel is plotted at every memory level at once, so no
  // single level owns the AI axis.
  function isSingleKernelIsolated() {
    return state.selected.size === 1;
  }

  // The memory level that currently owns the AI axis, or ALL_PEAKS_VALUE when
  // no single level does.
  function effectivePeak() {
    return isSingleKernelIsolated() ? ALL_PEAKS_VALUE : state.peak;
  }

  // ===== Runtime-percent filter ============================================

  function computeRuntimeBreakpoints() {
    kernelCumulativePct = {};
    runtimeBreakpoints = [];
    if (!hasRuntimeData) {
      return;
    }
    var order = kernelIndicesByRuntime();
    var cumulative = 0;
    var i = 0;
    while (i < order.length) {
      var pct = kernels[order[i]].pctRuntime || 0;
      var group = [];
      while (i < order.length && (kernels[order[i]].pctRuntime || 0) === pct) {
        group.push(order[i]);
        i += 1;
      }
      group.forEach(function (idx) {
        cumulative += kernels[idx].pctRuntime || 0;
      });
      group.forEach(function (idx) {
        kernelCumulativePct[idx] = cumulative;
      });
      runtimeBreakpoints.push(cumulative);
    }
  }

  function withinThreshold(kernel) {
    if (!hasRuntimeData) {
      return true;
    }
    return (kernelCumulativePct[kernel.index] || 0) <= state.runtimeThreshold + RUNTIME_EPSILON;
  }

  function kernelIsVisible(kernel) {
    if (!withinThreshold(kernel)) {
      return false;
    }
    return state.selected.size === 0 || state.selected.has(kernel.index);
  }

  function kernelIsDrawn(kernel) {
    return kernelIsVisible(kernel) && pointsForCurrentPeak(kernel).length > 0;
  }

  function isSoleSelected(kernel) {
    return isSingleKernelIsolated() && state.selected.has(kernel.index);
  }

  function pointsForCurrentPeak(kernel) {
    var peak = effectivePeak();
    if (peak === ALL_PEAKS_VALUE) {
      return kernel.points;
    }
    return kernel.points.filter(function (point) {
      return point.peak === peak;
    });
  }

  // ===== Compute ceilings / roof isolation =================================

  // Compute ceilings meet the diagonal at the leftmost bandwidth among the
  // isolated roofs, falling back to every roof when none of them has a
  // bandwidth to offer.
  function referenceBandwidth() {
    var bws = bandwidthsOf(
      rooflineTraces.filter(function (roof) {
        return state.isolatedRoofs.has(roof.traceIndex);
      })
    );
    if (!bws.length) {
      bws = bandwidthsOf(rooflineTraces);
    }
    return bws.length ? Math.max.apply(null, bws) : 0;
  }

  function bandwidthsOf(roofs) {
    return roofs
      .map(function (roof) {
        return roof.bandwidth;
      })
      .filter(function (bw) {
        return bw > 0;
      });
  }

  // Highlight overlays are shown only while isolating.
  function updateCeilings() {
    if (!plotlyReady() || !computeOverlayTraces.length) {
      return;
    }
    var isolating = state.isolatedRoofs.size > 0;
    var refBw = referenceBandwidth();
    var indices = [];
    var xs = [];
    var ys = [];
    var visibility = [];
    computeOverlayTraces.forEach(function (overlay) {
      indices.push(overlay.traceIndex);
      if (isolating && refBw) {
        // The overlay is a horizontal line with hover disabled, so its two
        // endpoints draw the same stroke that intermediate vertices would.
        var left = overlay.peakPerf / refBw;
        xs.push([left, ROOF_EXTREME_MAX_AI]);
        ys.push([overlay.peakPerf, overlay.peakPerf]);
        visibility.push(true);
      } else {
        xs.push([]);
        ys.push([]);
        visibility.push(false);
      }
    });
    Plotly.restyle(gd, { x: xs, y: ys, visible: visibility }, indices);
  }

  // Isolate the clicked memory roof(s) by dimming the others.
  function applyRoofIsolation() {
    if (!plotlyReady()) {
      return;
    }
    var isolating = state.isolatedRoofs.size > 0;
    var indices = [];
    var opacities = [];
    memoryRoofIndices.forEach(function (idx) {
      indices.push(idx);
      opacities.push(
        !isolating || state.isolatedRoofs.has(idx) ? 1 : PLOT_DIM_OPACITY
      );
    });
    computeCeilingIndices.forEach(function (idx) {
      indices.push(idx);
      opacities.push(isolating ? PLOT_DIM_OPACITY : 1);
    });
    if (indices.length) {
      Plotly.restyle(gd, { opacity: opacities }, indices);
    }
    applyRoofEmphasis();
    updateCeilings();
  }

  // Undefined until the first call: emphasized is only ever a level or null.
  var lastEmphasizedLevel;

  function applyRoofEmphasis() {
    if (!plotlyReady() || !memoryRoofIndices.length) {
      return;
    }
    // While roofs are isolated no level owns the AI axis, so none is thickened.
    var emphasized = state.isolatedRoofs.size > 0 ? null : effectivePeak();
    // A restyle here redraws the whole chart, and render() calls this on every
    // frame of a slider drag, so skip the ones that would change nothing.
    if (emphasized === lastEmphasizedLevel) {
      return;
    }
    lastEmphasizedLevel = emphasized;
    var widths = rooflineTraces.map(function (roof) {
      return roof.level === emphasized ? 4 : 2;
    });
    Plotly.restyle(gd, { "line.width": widths }, memoryRoofIndices);
  }

  // Isolate a memory roof, shared by the roofline panel rows and by clicking a
  // slope in the plot.
  function isolateRoof(traceIndex, multi) {
    if (memoryRoofIndices.indexOf(traceIndex) < 0) {
      return;
    }
    toggleSelection(state.isolatedRoofs, traceIndex, multi);
    applyRoofIsolation();
    updateRoofPanel();
  }

  // ===== Theme =============================================================

  function themeColor(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
  }

  function readPlotTheme() {
    return {
      paper: themeColor("--roofline-surface"),
      area: themeColor("--roofline-plot-area"),
      grid: themeColor("--roofline-plot-grid"),
      text: themeColor("--roofline-text"),
      // Hover cards and the exported legend float above the chart the way the
      // side panels sit beside it, so they borrow the panels' colors.
      overlay: themeColor("--roofline-bg-soft"),
      overlayBorder: themeColor("--roofline-border"),
      markerOutline: themeColor("--roofline-marker-outline"),
    };
  }

  // Repaint the chart in the active theme. Tick labels, axis titles, and the
  // chart title all inherit layout.font, so one color covers every label.
  function applyPlotTheme() {
    if (!plotlyReady()) {
      return;
    }
    var theme = readPlotTheme();
    Plotly.relayout(gd, {
      paper_bgcolor: theme.paper,
      plot_bgcolor: theme.area,
      "font.color": theme.text,
      "xaxis.gridcolor": theme.grid,
      "yaxis.gridcolor": theme.grid,
      "hoverlabel.bgcolor": theme.overlay,
      "hoverlabel.bordercolor": theme.overlayBorder,
      "hoverlabel.font.color": theme.text,
    });
    if (kernelTraceIndices.length) {
      // Kernel dots carry an outline so a kernel whose palette color lands near
      // the background is still legible against it.
      Plotly.restyle(
        gd,
        { "marker.line.color": theme.markerOutline },
        kernelTraceIndices
      );
    }
  }

  function themeIsDark() {
    return document.documentElement.classList.contains("roofline-theme-dark");
  }

  // The button names the theme it switches to, the way a light switch is
  // labeled by what it does rather than by where it is.
  function syncThemeToggle() {
    if (!themeToggleBtn) {
      return;
    }
    var dark = themeIsDark();
    themeToggleBtn.textContent = dark ? "Light mode" : "Dark mode";
    themeToggleBtn.setAttribute("aria-pressed", String(dark));
    themeToggleBtn.title = dark
      ? "Switch the page and chart to light colors"
      : "Switch the page and chart to dark colors";
  }

  function setTheme(dark) {
    document.documentElement.classList.toggle("roofline-theme-dark", dark);
    syncThemeToggle();
    applyPlotTheme();
  }

  function watchSystemTheme() {
    if (typeof window.matchMedia !== "function") {
      return;
    }
    var query = window.matchMedia("(prefers-color-scheme: dark)");
    var onChange = function (event) {
      if (!themeIsReaderChoice) {
        setTheme(event.matches);
      }
    };
    if (typeof query.addEventListener === "function") {
      query.addEventListener("change", onChange);
    } else if (typeof query.addListener === "function") {
      query.addListener(onChange);
    }
  }

  // ===== View framing ======================================================
  function frameAnchors() {
    var xs = [];
    var ys = [];
    kernels.forEach(function (kernel) {
      if (!kernelIsDrawn(kernel)) {
        return;
      }
      pointsForCurrentPeak(kernel).forEach(function (point) {
        if (point.ai > 0 && point.perf > 0) {
          xs.push(point.ai);
          ys.push(point.perf);
        }
      });
    });
    if (!xs.length) {
      return null;
    }
    rooflineTraces.forEach(function (roof) {
      if (roof.kneeAi > 0 && roof.kneePerf > 0) {
        xs.push(roof.kneeAi);
        ys.push(roof.kneePerf);
      }
    });
    computeTraces.forEach(function (ceiling) {
      if (ceiling.peakPerf > 0) {
        ys.push(ceiling.peakPerf);
      }
    });
    var perfLo = Math.min.apply(null, ys);
    rooflineTraces.forEach(function (roof) {
      if (roof.bandwidth > 0) {
        xs.push(perfLo / roof.bandwidth);
      }
    });
    return { xs: xs, ys: ys };
  }

  function paddedLogSpan(lo, hi) {
    return [
      Math.log10(lo) - Math.log10(FRAME_PAD),
      Math.log10(hi) + Math.log10(FRAME_PAD),
    ];
  }

  function widenTo(range, decades) {
    var span = range[1] - range[0];
    if (!(decades > span)) {
      return range.slice();
    }
    var mid = 0.5 * (range[0] + range[1]);
    return [mid - 0.5 * decades, mid + 0.5 * decades];
  }

  // Pixel size of the plotting area
  function plotAreaPixels() {
    var margin = (gd.layout && gd.layout.margin) || {};
    var width = gd.clientWidth - (margin.l || 0) - (margin.r || 0);
    var height = gd.clientHeight - (margin.t || 0) - (margin.b || 0);
    if (!(width > 0) || !(height > 0)) {
      return null;
    }
    return { width: width, height: height };
  }

  function shapeToPlotArea(frame) {
    var area = plotAreaPixels();
    if (!area) {
      return frame;
    }
    var xSpan = frame.x[1] - frame.x[0];
    var ySpan = frame.y[1] - frame.y[0];
    if (!(xSpan > 0) || !(ySpan > 0)) {
      return frame;
    }
    var screenSlope = (area.height * xSpan) / (area.width * ySpan);
    if (screenSlope > FRAME_SLOPE_SKEW) {
      return {
        x: frame.x.slice(),
        y: widenTo(
          frame.y,
          (area.height * xSpan) / (area.width * FRAME_SLOPE_SKEW)
        ),
      };
    }
    if (screenSlope < 1 / FRAME_SLOPE_SKEW) {
      return {
        x: widenTo(
          frame.x,
          (area.width * ySpan) / (FRAME_SLOPE_SKEW * area.height)
        ),
        y: frame.y.slice(),
      };
    }
    return frame;
  }

  function pinToSlopes(frame) {
    var slopes = rooflineTraces
      .map(function (roof) {
        return Math.log10(roof.bandwidth);
      })
      .filter(function (slope) {
        return isFinite(slope);
      });
    if (!slopes.length) {
      return frame;
    }
    var xLo = Math.min(frame.x[0], frame.y[0] - Math.max.apply(null, slopes));
    var area = plotAreaPixels();
    var y = frame.y.slice();
    if (area) {
      var roomForSlope =
        (area.height * (frame.x[1] - xLo)) / (area.width * FRAME_SLOPE_SKEW);
      if (roomForSlope > y[1] - y[0]) {
        y = [y[0], y[0] + roomForSlope];
      }
    }
    return { x: [xLo, frame.x[1]], y: y };
  }

  // Log-axis frame for what is drawn right now. Null when nothing is drawn, so
  // the caller can fall back to the range the figure was built with.
  function currentFrame() {
    var anchors = frameAnchors();
    if (!anchors) {
      return null;
    }
    var frame = {
      x: paddedLogSpan(
        Math.min.apply(null, anchors.xs),
        Math.max.apply(null, anchors.xs)
      ),
      y: paddedLogSpan(
        Math.min.apply(null, anchors.ys),
        Math.max.apply(null, anchors.ys)
      ),
    };
    frame.x = widenTo(frame.x, FRAME_MIN_DECADES);
    return pinToSlopes(shapeToPlotArea(frame));
  }

  // Pin the axes to a computed frame
  function applyFrame(frame) {
    applyingFrame = true;
    autoFramed = true;
    var settled = Plotly.relayout(gd, {
      "xaxis.range": frame.x,
      "yaxis.range": frame.y,
    });
    var release = function () {
      applyingFrame = false;
    };
    if (settled && typeof settled.then === "function") {
      settled.then(release, release);
    } else {
      release();
    }
  }

  // Reset (button, double-click, and the opening view): re-frame on whatever
  // kernels are currently shown, so it follows the active filter and selection
  // instead of a fixed spot.
  function resetView() {
    if (!plotlyReady()) {
      return;
    }
    var frame = currentFrame() || initialRange;
    if (!frame) {
      return;
    }
    applyFrame(frame);
  }

  // ===== PNG export ========================================================

  function exportTextWidth(text) {
    if (!exportTextMeasureContext) {
      exportTextMeasureContext = document
        .createElement("canvas")
        .getContext("2d");
      // Only kernel-legend labels/title are measured here, and the kernel
      // legend is monospace, so measure with the kernel-name font.
      exportTextMeasureContext.font =
        EXPORT_LEGEND_FONT_SIZE + "px " + KERNEL_NAME_FONT_FAMILY;
    }
    return exportTextMeasureContext.measureText(text).width;
  }

  function textPrefixLength(text, maximumWidth) {
    var lowerBound = 0;
    var upperBound = text.length;
    while (lowerBound < upperBound) {
      var midpoint = Math.ceil((lowerBound + upperBound) / 2);
      if (exportTextWidth(text.slice(0, midpoint)) <= maximumWidth) {
        lowerBound = midpoint;
      } else {
        upperBound = midpoint - 1;
      }
    }
    return lowerBound;
  }

  function fitTextToWidth(text, maximumWidth) {
    if (exportTextWidth(text) <= maximumWidth) {
      return text;
    }

    var ellipsis = "\u2026";
    var prefixWidth = Math.max(0, maximumWidth - exportTextWidth(ellipsis));
    return text.slice(0, textPrefixLength(text, prefixWidth)) + ellipsis;
  }

  function preferredWrapLength(text, maximumLength) {
    var minimumPreferredLength = Math.floor(maximumLength * 0.55);
    for (
      var length = maximumLength;
      length > minimumPreferredLength;
      length--
    ) {
      if (/[\s_,;:>)]/.test(text.charAt(length - 1))) {
        return length;
      }
    }
    return maximumLength;
  }

  function wrapTextToWidth(text, maximumWidth, maximumLines, finalSuffix) {
    var lines = [];
    var remainingText = text;

    for (var lineIndex = 0; lineIndex < maximumLines; lineIndex++) {
      if (exportTextWidth(remainingText + finalSuffix) <= maximumWidth) {
        lines.push(remainingText + finalSuffix);
        break;
      }

      var isFinalLine = lineIndex === maximumLines - 1;
      if (isFinalLine) {
        var finalTextWidth = Math.max(
          0,
          maximumWidth - exportTextWidth(finalSuffix)
        );
        lines.push(fitTextToWidth(remainingText, finalTextWidth) + finalSuffix);
        break;
      }

      var fittedLength = textPrefixLength(remainingText, maximumWidth);
      var wrapLength = preferredWrapLength(
        remainingText,
        Math.max(1, fittedLength)
      );
      lines.push(remainingText.slice(0, wrapLength));
      remainingText = remainingText.slice(wrapLength).replace(/^\s+/, "");
    }

    return lines.join("<br>");
  }

  function kernelExportRuntimeSuffix(kernel) {
    if (!kernelHasRuntime(kernel)) {
      return "";
    }
    return "   " + kernel.pctRuntime.toFixed(2) + "%";
  }

  function exportLegendLayout(x, xAnchor, y, yAnchor, fontFamily) {
    var theme = readPlotTheme();
    return {
      x: x,
      xanchor: xAnchor,
      y: y,
      yanchor: yAnchor,
      bgcolor: theme.overlay,
      bordercolor: theme.overlayBorder,
      borderwidth: 1,
      font: {
        size: EXPORT_LEGEND_FONT_SIZE,
        family: fontFamily,
        color: theme.text,
      },
      itemclick: false,
      itemdoubleclick: false,
    };
  }

  function exportKernelLegendTitle(visibleKernelCount) {
    return "Kernels " + formatCount(visibleKernelCount, kernels.length);
  }

  function exportKernelLegendWidth(visibleKernels, plotWidth) {
    var title = exportKernelLegendTitle(visibleKernels.length);
    var naturalWidth = exportTextWidth(title) + EXPORT_LEGEND_TEXT_INSET;
    visibleKernels.forEach(function (entry) {
      var fullLabel =
        entry.kernel.name + kernelExportRuntimeSuffix(entry.kernel);
      naturalWidth = Math.max(
        naturalWidth,
        exportTextWidth(fullLabel) + EXPORT_LEGEND_TEXT_INSET
      );
    });

    var responsiveMaximum = clamp(
      plotWidth * EXPORT_LEGEND_WIDTH_RATIO,
      EXPORT_LEGEND_MIN_WIDTH,
      EXPORT_LEGEND_MAX_WIDTH
    );
    return clamp(
      naturalWidth,
      EXPORT_LEGEND_MIN_WIDTH,
      responsiveMaximum
    );
  }

  // Legend rows the figure can hold without growing past
  // EXPORT_LEGEND_MAX_HEIGHT_RATIO times the plot. This is what actually bounds
  // the exported canvas: without it the figure grows one row per kernel until
  // the browser silently hands back a blank image.
  function exportLegendRowCapacity(plotHeight) {
    return Math.max(
      1,
      Math.floor(
        (plotHeight * EXPORT_LEGEND_MAX_HEIGHT_RATIO -
          EXPORT_LEGEND_HEADER_HEIGHT) /
          EXPORT_LEGEND_ROW_HEIGHT
      ) - 1
    );
  }

  function exportKernelLabelLines(kernelCount, plotHeight) {
    if (!kernelCount) {
      return 1;
    }
    return clamp(
      Math.floor(exportLegendRowCapacity(plotHeight) / kernelCount),
      1,
      EXPORT_LEGEND_MAX_LABEL_LINES
    );
  }

  function buildExportDimensions(visibleKernels) {
    var chartWidth = gd.clientWidth || EXPORT_MIN_WIDTH;
    var chartHeight = gd.clientHeight || EXPORT_MIN_HEIGHT;
    var scale = Math.max(
      1,
      EXPORT_MIN_WIDTH / chartWidth,
      EXPORT_MIN_HEIGHT / chartHeight
    );
    var plotWidth = Math.round(chartWidth * scale);
    var plotHeight = Math.round(chartHeight * scale);
    var hasKernelLegend = visibleKernels.length > 0;
    var legendWidth = hasKernelLegend
      ? exportKernelLegendWidth(visibleKernels, plotWidth)
      : 0;
    var kernelLabelLines = exportKernelLabelLines(
      visibleKernels.length,
      plotHeight
    );
    // Kernels past the capacity are summarized in one final row instead of
    // being clipped away without a trace.
    var kernelLegendLimit = Math.max(
      1,
      Math.floor(exportLegendRowCapacity(plotHeight) / kernelLabelLines)
    );
    var listedKernels = Math.min(visibleKernels.length, kernelLegendLimit);
    var overflowRows = listedKernels < visibleKernels.length ? 1 : 0;
    var legendHeight =
      EXPORT_LEGEND_HEADER_HEIGHT +
      (listedKernels * kernelLabelLines + overflowRows + 1) *
        EXPORT_LEGEND_ROW_HEIGHT;

    return {
      width: plotWidth + legendWidth,
      height: hasKernelLegend
        ? Math.max(plotHeight, legendHeight)
        : plotHeight,
      legendWidth: legendWidth,
      legendTextWidth: Math.max(
        0,
        legendWidth - EXPORT_LEGEND_TEXT_INSET
      ),
      hasKernelLegend: hasKernelLegend,
      kernelLabelLines: kernelLabelLines,
      kernelLegendLimit: kernelLegendLimit,
    };
  }

  // Raster resolution for the download, reduced from the display-matching scale
  // only as far as the canvas limits demand.
  function exportRasterScale(width, height) {
    var requested = clamp(window.devicePixelRatio || 2, 2, EXPORT_MAX_SCALE);
    var longestSide = Math.max(width, height, 1);
    var area = Math.max(width * height, 1);
    return Math.max(
      1,
      Math.min(
        requested,
        EXPORT_MAX_RASTER_DIMENSION / longestSide,
        Math.sqrt(EXPORT_MAX_RASTER_AREA / area)
      )
    );
  }

  // Turn traces the interactive page keeps out of the legend into the rows of
  // one export legend, titled on its first row the way Plotly groups them.
  function listInExportLegend(data, entries, spec) {
    entries.forEach(function (entry, row) {
      var trace = data[entry.traceIndex];
      trace.showlegend = true;
      trace.name = spec.label(entry);
      trace.legend = spec.legend;
      trace.legendgroup = spec.group;
      trace.legendrank = spec.rankFrom + row;
      if (row === 0) {
        trace.legendgrouptitle = { text: spec.title };
      }
    });
  }

  // Build an export-only Plotly figure
  function buildExportFigure() {
    var data = JSON.parse(JSON.stringify(gd.data));
    var layout = JSON.parse(JSON.stringify(gd.layout));
    pinExportAxisRanges(layout);
    data.forEach(function (trace) {
      trace.showlegend = false;
    });

    // Heaviest kernel first
    var visibleKernels = [];
    kernelIndicesByRuntime().forEach(function (position) {
      var kernel = kernels[position];
      if (!kernelIsDrawn(kernel)) {
        return;
      }
      visibleKernels.push({
        kernel: kernel,
        traceIndex: kernelTraceIndices[position],
      });
    });
    var dimensions = buildExportDimensions(visibleKernels);
    var legendKernels = visibleKernels.slice(0, dimensions.kernelLegendLimit);

    listInExportLegend(data, legendKernels, {
      legend: "legend",
      group: EXPORT_KERNEL_LEGEND_GROUP,
      title: exportKernelLegendTitle(visibleKernels.length),
      rankFrom: 0,
      label: function (entry) {
        return wrapTextToWidth(
          entry.kernel.name,
          dimensions.legendTextWidth,
          dimensions.kernelLabelLines,
          kernelExportRuntimeSuffix(entry.kernel)
        );
      },
    });

    var roofEntries = exportRoofLegendEntries();
    listInExportLegend(data, roofEntries, {
      legend: "legend2",
      group: "export-roofs",
      title: "Rooflines",
      // Ranked below every kernel, so the two legends stack in a fixed order
      // however many kernels are listed.
      rankFrom: EXPORT_ROOF_LEGEND_RANK,
      label: function (entry) {
        return entry.label;
      },
    });

    if (legendKernels.length < visibleKernels.length) {
      // A legend-only trace standing in for the kernels that did not fit, so a
      // truncated legend never reads as the complete set.
      data.push({
        type: "scatter",
        x: [null],
        y: [null],
        mode: "markers",
        marker: { color: FALLBACK_COLOR, size: 0, opacity: 0 },
        hoverinfo: "skip",
        showlegend: true,
        legend: "legend",
        legendgroup: EXPORT_KERNEL_LEGEND_GROUP,
        legendrank: legendKernels.length,
        name:
          "+" +
          (visibleKernels.length - legendKernels.length) +
          " more drawn but not listed",
      });
    }

    layout.autosize = false;
    layout.width = dimensions.width;
    layout.height = dimensions.height;
    layout.showlegend = legendKernels.length > 0 || roofEntries.length > 0;
    layout.hovermode = false;
    layout.dragmode = false;
    layout.margin = layout.margin || {};
    if (dimensions.hasKernelLegend) {
      layout.margin.r =
        (layout.margin.r || 0) + dimensions.legendWidth;
    }
    // The PNG is the shareable artifact, so it has to say which view it is.
    layout.margin.t = (layout.margin.t || 0) + EXPORT_SUBTITLE_HEIGHT;
    applyExportSubtitle(layout);
    layout.legend = exportLegendLayout(
      1.02,
      "left",
      1,
      "top",
      KERNEL_NAME_FONT_FAMILY
    );
    layout.legend.tracegroupgap = 12;
    layout.legend2 = exportLegendLayout(
      0.99,
      "right",
      0.01,
      "bottom",
      EXPORT_LEGEND_FONT_FAMILY
    );

    return {
      data: data,
      layout: layout,
      width: dimensions.width,
      height: dimensions.height,
    };
  }

  // Freeze the live view
  function pinExportAxisRanges(layout) {
    ["xaxis", "yaxis"].forEach(function (axisName) {
      var axis = layout[axisName];
      if (!axis || !axis.range) {
        return;
      }
      axis.range = axis.range.slice();
      axis.autorange = false;
    });
  }

  // Every roof drawn, memory slopes then compute ceilings, so the exported
  // horizontal lines are labeled instead of anonymous.
  function exportRoofLegendEntries() {
    var entries = rooflineTraces.map(function (roof) {
      return { traceIndex: roof.traceIndex, label: roof.level };
    });
    computeTraces.forEach(function (ceiling) {
      if (ceiling.label) {
        entries.push({
          traceIndex: ceiling.traceIndex,
          label: ceiling.label,
        });
      }
    });
    return entries;
  }

  function exportViewSubtitle() {
    var peak = effectivePeak();
    var parts = [
      "AI axis: " + (peak === ALL_PEAKS_VALUE ? ALL_PEAKS_LABEL : peak),
    ];
    if (hasRuntimeData && isFinite(state.runtimeThreshold)) {
      parts.push("runtime shown: " + state.runtimeThreshold.toFixed(3) + "%");
    }
    if (state.isolatedRoofs.size > 0) {
      parts.push("rooflines isolated: " + state.isolatedRoofs.size);
    }
    return parts.join("  \u00b7  ");
  }

  function applyExportSubtitle(layout) {
    var title = layout.title || {};
    var text = title.text || "";
    layout.title = {
      x: title.x != null ? title.x : 0.5,
      xanchor: title.xanchor || "center",
      font: title.font,
      text:
        text +
        '<br><span style="font-size:' +
        EXPORT_LEGEND_FONT_SIZE +
        'px">' +
        exportViewSubtitle() +
        "</span>",
    };
  }

  // Rasterize the export-only figure at high resolution
  function exportPng() {
    if (
      !plotlyReady() ||
      typeof Plotly.downloadImage !== "function" ||
      !exportPngBtn
    ) {
      return;
    }

    var previousLabel = exportPngBtn.textContent;
    exportPngBtn.disabled = true;
    exportPngBtn.textContent = "Exporting...";

    var exportGraph = null;
    function finish() {
      if (exportGraph) {
        Plotly.purge(exportGraph);
        exportGraph.remove();
        exportGraph = null;
      }
      exportPngBtn.disabled = false;
      exportPngBtn.textContent = previousLabel;
    }

    try {
      var fileName = (document.title || "roofline")
        .replace(/[^A-Za-z0-9._-]+/g, "_")
        .replace(/^_+|_+$/g, "");
      var figure = buildExportFigure();
      exportGraph = document.createElement("div");
      exportGraph.style.position = "absolute";
      exportGraph.style.left = "-100000px";
      exportGraph.style.top = "0";
      exportGraph.style.width = figure.width + "px";
      exportGraph.style.height = figure.height + "px";
      document.body.appendChild(exportGraph);

      Plotly.newPlot(exportGraph, figure.data, figure.layout, {
        displayModeBar: false,
        responsive: false,
        staticPlot: true,
      })
        .then(function () {
          return Plotly.downloadImage(exportGraph, {
            format: "png",
            filename: fileName || "roofline",
            width: figure.width,
            height: figure.height,
            scale: exportRasterScale(figure.width, figure.height),
          });
        })
        .then(finish, function (error) {
          console.error("PNG export failed:", error);
          finish();
        });
    } catch (error) {
      console.error("PNG export failed:", error);
      finish();
    }
  }

  // ===== Kernel rendering ==================================================

  function syncPeakControl() {
    if (!peakSelect) {
      return;
    }
    var isolatedSingle = isSingleKernelIsolated();
    peakSelect.value = effectivePeak();
    peakSelect.disabled = isolatedSingle;
    if (peakControl) {
      peakControl.title = isolatedSingle
        ? "Locked while one kernel is isolated: that kernel is plotted at "
          + "every memory level at once, so no single level owns the AI axis."
        : peakControlTitle;
    }
  }

  function kernelPointColors(kernel, points) {
    var base = kernel.color || FALLBACK_COLOR;
    var byLevel = isSoleSelected(kernel);
    return points.map(function (point) {
      return byLevel ? peakColors[point.peak] || base : base;
    });
  }

  // Build the per-kernel Plotly restyle payload for the current peak/selection.
  function buildKernelRestylePayload() {
    var xs = [];
    var ys = [];
    var markerColors = [];
    var customdata = [];
    var visibility = [];

    kernels.forEach(function (kernel) {
      var visible = kernelIsVisible(kernel);
      var points = visible ? pointsForCurrentPeak(kernel) : [];
      xs.push(
        points.map(function (point) {
          return point.ai;
        })
      );
      ys.push(
        points.map(function (point) {
          return point.perf;
        })
      );
      markerColors.push(kernelPointColors(kernel, points));
      // Each trace's hovertemplate holds the kernel-level text; these are the
      // two values that vary per point, already formatted server-side.
      customdata.push(
        points.map(function (point) {
          return point.hoverCells;
        })
      );
      visibility.push(visible && points.length > 0);
    });

    return {
      xs: xs,
      ys: ys,
      markerColors: markerColors,
      customdata: customdata,
      visibility: visibility,
    };
  }

  function render() {
    syncPeakControl();
    if (plotlyReady() && kernelTraceIndices.length) {
      var payload = buildKernelRestylePayload();
      Plotly.restyle(
        gd,
        {
          x: payload.xs,
          y: payload.ys,
          "marker.color": payload.markerColors,
          customdata: payload.customdata,
          visible: payload.visibility,
        },
        kernelTraceIndices
      );
      applyRoofEmphasis();
    }
    updatePanel();
    updateRoofPanel();
  }

  // Coalesce repaints driven by a continuous control onto one animation frame.
  function scheduleRender() {
    if (renderFrame != null) {
      return;
    }
    renderFrame = window.requestAnimationFrame(function () {
      renderFrame = null;
      render();
    });
  }

  function toggleKernel(index, event) {
    var multi = isMultiSelectEvent(event);
    if (multi && state.selected.size === 0) {
      kernels.forEach(function (kernel) {
        state.selected.add(kernel.index);
      });
      state.selected.delete(index);
    } else {
      toggleSelection(state.selected, index, multi);
    }
    render();
    // Isolating a single kernel brings its row into view in the kernel table.
    if (isSingleKernelIsolated()) {
      scrollKernelIntoView(index);
    }
  }

  // Scroll a kernel's row into view within the kernel list.
  function scrollKernelIntoView(index) {
    eachKernelRow(function (item, kernel) {
      if (kernel.index === index) {
        item.scrollIntoView({ block: "nearest" });
      }
    });
  }

  // ===== Panels ============================================================

  // Build one panel with a color swatch, a label, and optional trailing
  // nodes. Shared by the kernel and roofline panels.
  function createPanelRow(opts) {
    var item = document.createElement("li");
    item.className = "roofline-panel-item";
    Object.keys(opts.dataset).forEach(function (key) {
      item.dataset[key] = opts.dataset[key];
    });

    var swatch = document.createElement("span");
    swatch.className = opts.swatchClass || "roofline-swatch";
    swatch.style.backgroundColor = opts.color || FALLBACK_COLOR;

    var label = document.createElement("span");
    label.className = opts.labelClass || "roofline-panel-name";
    label.textContent = opts.label;

    item.appendChild(swatch);
    item.appendChild(label);
    opts.extras.forEach(function (node) {
      item.appendChild(node);
    });
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.addEventListener("click", opts.onClick);
    item.addEventListener("keydown", function (event) {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        opts.onClick(event);
      }
    });
    return item;
  }

  function buildPeakOptions() {
    if (!peakSelect) {
      return;
    }
    model.peaks.forEach(function (peak) {
      var el = document.createElement("option");
      el.value = peak;
      el.textContent = peak;
      peakSelect.appendChild(el);
    });
    var allEl = document.createElement("option");
    allEl.value = ALL_PEAKS_VALUE;
    allEl.textContent = ALL_PEAKS_LABEL;
    peakSelect.appendChild(allEl);
    peakSelect.value = state.peak;
  }

  function buildKernelPanel() {
    if (!kernelList) {
      return;
    }
    // Show the heaviest kernels first.
    kernelIndicesByRuntime().forEach(function (index) {
      var kernel = kernels[index];
      var extras = [];
      if (kernelHasRuntime(kernel)) {
        var pct = document.createElement("span");
        pct.className = "roofline-kernel-pct";
        pct.textContent = kernel.pctRuntime.toFixed(1) + "%";
        pct.title = "Percent of GPU resident time";
        extras.push(pct);
      }
      kernelList.appendChild(
        createPanelRow({
          color: kernel.color,
          label: kernel.name,
          labelClass: "roofline-panel-name roofline-kernel-name",
          dataset: { index: String(index) },
          extras: extras,
          onClick: function (event) {
            toggleKernel(kernel.index, event);
          },
        })
      );
    });
  }

  function buildRoofPanel() {
    if (!roofList) {
      return;
    }
    rooflineTraces.forEach(function (roof) {
      var aiaxis = document.createElement("span");
      aiaxis.className = "roofline-roof-aiaxis";
      roofList.appendChild(
        createPanelRow({
          color: peakColors[roof.level] || FALLBACK_COLOR,
          label: roof.level,
          swatchClass: "roofline-swatch roofline-roof-swatch",
          dataset: { trace: String(roof.traceIndex), level: roof.level },
          extras: [aiaxis],
          onClick: function (event) {
            isolateRoof(roof.traceIndex, isMultiSelectEvent(event));
          },
        })
      );
    });
  }

  // Reflect isolation state and the active region in the roofline
  // panel rows, plus the "(shown / total)" count and the reset button.
  function updateRoofPanel() {
    var isolating = state.isolatedRoofs.size > 0;
    var axisPeak = effectivePeak();
    if (roofList) {
      Array.prototype.forEach.call(roofList.children, function (item) {
        var idx = Number(item.dataset.trace);
        var isolated = state.isolatedRoofs.has(idx);
        setRowState(item, isolated, isolating && !isolated);
        var aiaxis = item.querySelector(".roofline-roof-aiaxis");
        if (aiaxis) {
          aiaxis.textContent = item.dataset.level === axisPeak ? "(AI axis)" : "";
        }
      });
    }
    if (roofCountEl) {
      var total = rooflineTraces.length;
      var shown = isolating ? state.isolatedRoofs.size : total;
      roofCountEl.textContent = formatCount(shown, total);
    }
    if (showAllRoofsBtn) {
      showAllRoofsBtn.disabled = !isolating;
    }
  }

  // Hard-stop linear gradient so each memory-level color shows as its own band.
  function swatchGradient(colors) {
    var count = colors.length;
    var stops = colors.map(function (color, i) {
      var start = ((i / count) * 100).toFixed(2);
      var end = (((i + 1) / count) * 100).toFixed(2);
      return color + " " + start + "%, " + color + " " + end + "%";
    });
    return "linear-gradient(90deg, " + stops.join(", ") + ")";
  }

  function updatePanel() {
    var filtering = state.selected.size > 0;
    eachKernelRow(function (item, kernel) {
      var selected = state.selected.has(kernel.index);
      setRowState(item, selected, filtering && !selected);
      item.classList.toggle("filtered", !withinThreshold(kernel));
      var swatch = item.querySelector(".roofline-swatch");
      if (swatch) {
        var colors = kernelPointColors(kernel, kernel.points);
        var banded = colors.some(function (color) {
          return color !== colors[0];
        });
        swatch.style.background = banded
          ? swatchGradient(colors)
          : colors[0] || FALLBACK_COLOR;
      }
    });
    if (kernelCountEl) {
      kernelCountEl.textContent = formatCount(
        kernels.filter(kernelIsDrawn).length,
        kernels.length
      );
    }
    if (showAllBtn) {
      showAllBtn.disabled = !filtering;
    }
  }

  // ===== Wiring / lifecycle ================================================

  function wireEvents() {
    if (peakSelect) {
      peakSelect.addEventListener("change", function () {
        // "all" is a valid, user-selectable region (every memory level).
        state.peak = peakSelect.value;
        render();
      });
    }
    if (showAllBtn) {
      showAllBtn.addEventListener("click", function () {
        state.selected.clear();
        render();
      });
    }
    if (showAllRoofsBtn) {
      showAllRoofsBtn.addEventListener("click", function () {
        state.isolatedRoofs.clear();
        applyRoofIsolation();
        updateRoofPanel();
      });
    }
    if (runtimeSlider) {
      runtimeSlider.addEventListener("input", function () {
        state.runtimeThreshold = runtimeBreakpoints[Number(runtimeSlider.value)];
        updateRuntimeLabel();
        scheduleRender();
      });
    }
    if (resetViewBtn) {
      resetViewBtn.addEventListener("click", resetView);
    }
    if (exportPngBtn) {
      exportPngBtn.addEventListener("click", exportPng);
    }
    if (themeToggleBtn) {
      themeToggleBtn.addEventListener("click", function () {
        themeIsReaderChoice = true;
        setTheme(!themeIsDark());
      });
    }
    if (gd && typeof gd.on === "function") {
      gd.on("plotly_doubleclick", resetView);
      gd.on("plotly_relayout", function (payload) {
        if (applyingFrame || !payload) {
          return;
        }
        var movedAxes = Object.keys(payload).some(function (key) {
          return key.indexOf("axis.range") >= 0;
        });
        if (movedAxes) {
          autoFramed = false;
        }
      });
      gd.on("plotly_click", function (data) {
        if (!data || !data.points || !data.points.length) {
          return;
        }
        var traceIndex = data.points[0].curveNumber;
        // Clicking a roof slope isolates it, same as its panel row.
        if (memoryRoofIndices.indexOf(traceIndex) >= 0) {
          isolateRoof(traceIndex, isMultiSelectEvent(data.event));
          return;
        }
        var position = kernelTraceIndices.indexOf(traceIndex);
        if (position < 0 || !kernels[position]) {
          return;
        }
        toggleKernel(kernels[position].index, data.event);
      });
    }
  }

  function whenPlotReady(callback, attemptsLeft) {
    if (plotlyReady() && typeof gd.on === "function") {
      callback();
      return;
    }
    if (attemptsLeft <= 0) {
      callback();
      return;
    }
    setTimeout(function () {
      whenPlotReady(callback, attemptsLeft - 1);
    }, PLOT_READY_POLL_MS);
  }

  function resizePlot() {
    if (plotlyReady() && Plotly.Plots) {
      Plotly.Plots.resize(gd);
    }
  }

  function schedulePlotResize() {
    if (plotResizeFrame != null) {
      return;
    }
    plotResizeFrame = window.requestAnimationFrame(function () {
      plotResizeFrame = null;
      resizePlot();
      if (autoFramed) {
        resetView();
      }
    });
  }

  function observePlotContainer() {
    if (plotColumn && typeof window.ResizeObserver === "function") {
      new window.ResizeObserver(schedulePlotResize).observe(plotColumn);
    }
  }

  // Remember the baked initial log-axis range
  function captureInitialRange() {
    if (!gd || !gd.layout || !gd.layout.xaxis || !gd.layout.yaxis) {
      return;
    }
    var xr = gd.layout.xaxis.range;
    var yr = gd.layout.yaxis.range;
    if (xr && yr) {
      initialRange = { x: xr.slice(), y: yr.slice() };
    }
  }

  // Show the true cumulative percent of runtime covered at the current stop.
  function updateRuntimeLabel() {
    if (runtimeValueEl) {
      runtimeValueEl.textContent = state.runtimeThreshold.toFixed(3) + "%";
    }
  }

  // Point the slider at the data-driven breakpoints: one stop per kernel
  // boundary, defaulting to the last.
  function initRuntimeSlider() {
    if (!runtimeSlider || !runtimeBreakpoints.length) {
      return;
    }
    var lastIndex = runtimeBreakpoints.length - 1;
    runtimeSlider.min = "0";
    runtimeSlider.max = String(lastIndex);
    runtimeSlider.step = "1";
    runtimeSlider.value = String(lastIndex);
    state.runtimeThreshold = runtimeBreakpoints[lastIndex];
    updateRuntimeLabel();
  }

  function init() {
    // Seed the kernel-name font so the CSS picks it up for the panel labels.
    document.documentElement.style.setProperty(
      "--roofline-kernel-font",
      KERNEL_NAME_FONT_FAMILY
    );
    buildPeakOptions();
    buildKernelPanel();
    buildRoofPanel();
    computeRuntimeBreakpoints();
    // The runtime filter is meaningless without per-kernel runtime data.
    if (runtimeFilterEl && !hasRuntimeData) {
      runtimeFilterEl.style.display = "none";
    }
    initRuntimeSlider();
    syncThemeToggle();
    watchSystemTheme();
    whenPlotReady(function () {
      captureInitialRange();
      wireEvents();
      observePlotContainer();
      resizePlot();
      applyPlotTheme();
      render();
      resetView();
    }, PLOT_READY_MAX_ATTEMPTS);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
