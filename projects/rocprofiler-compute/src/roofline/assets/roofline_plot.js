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

  // ---- Config forwarded from roofline_shared.py via the model -------------
  var ALL_PEAKS_VALUE = model.allPeaksValue || "all";
  var ALL_PEAKS_LABEL = model.allPeaksLabel || "All peaks";
  var FALLBACK_COLOR = model.fallbackColor || "#888888";
  var KERNEL_NAME_FONT_FAMILY =
    model.kernelNameFontFamily ||
    "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  var ROOF_EXTREME_MAX_AI = model.roofExtremeMaxAi || 1e150;
  var PLOT_DIM_OPACITY =
    model.plotDimOpacity != null ? model.plotDimOpacity : 0.15;
  var ROOF_SAMPLES = model.roofSamples || 200;
  var FRAME_PAD = model.framePad;
  var FRAME_MIN_DECADES = model.frameMinDecades;
  var FRAME_ROOF_SEGMENT_DECADES = model.frameRoofSegmentDecades;
  var RUNTIME_EPSILON = 1e-6;
  // Memory-roof line widths: the AI-axis roof is drawn thicker for emphasis.
  var ROOF_WIDTH_NORMAL = 2;
  var ROOF_WIDTH_EMPHASIS = 4;
  // Preserve the current chart aspect ratio while ensuring publication-sized
  // output even when the browser viewport is small.
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
  var EXPORT_LEGEND_GLYPH_WIDTH_RATIO = 0.6;
  var EXPORT_ROOF_LEGEND_RANK = 10000;
  // Poll for Plotly to finish its initial paint before wiring interactivity.
  var PLOT_READY_POLL_MS = 50;
  var PLOT_READY_MAX_ATTEMPTS = 40;

  // ---- DOM handles --------------------------------------------------------
  var gd = document.getElementById(model.divId);
  var peakSelect = document.getElementById("roofline-peak-select");
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
  var plotColumn = gd ? gd.closest(".roofline-plot-col") : null;
  var plotResizeObserver = null;
  var plotResizeFrame = null;
  var exportTextMeasureContext = null;

  // ---- Model data ---------------------------------------------------------
  var kernels = model.kernels || [];
  // Each kernel's array position is its stable identity for selection,
  // scrolling, and runtime lookup. Keying on the display name would collapse
  // distinct kernels that share a trimmed signature.
  kernels.forEach(function (kernel, index) {
    kernel.index = index;
  });
  var kernelTraceIndices = model.kernelTraceIndices || [];
  var rooflineTraces = model.rooflineTraces || [];
  var computeTraces = model.computeTraces || [];
  var computeOverlayTraces = model.computeOverlayTraces || [];
  var peakColors = model.peakColors || {};
  var ceilingDenseHi = model.ceilingDenseHi || 0;
  var initialRange = null;

  // Whether any kernel carries a percent-of-runtime, which gates the filter.
  var hasRuntimeData = kernels.some(function (kernel) {
    return kernel.pctRuntime != null && isFinite(kernel.pctRuntime);
  });

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

  function setRowState(item, selected, dimmed) {
    item.classList.toggle("selected", selected);
    item.classList.toggle("dimmed", dimmed);
  }

  // Shared count shown next to each panel title.
  function formatCount(shown, total) {
    return "(" + shown + " / " + total + ")";
  }

  // Iterate the kernel-list rows, yielding the row element.
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

  function logspace(lo, hi, n) {
    var a = Math.log10(lo);
    var b = Math.log10(hi);
    var steps = Math.max(n - 1, 1);
    var out = [];
    for (var i = 0; i < n; i++) {
      out.push(Math.pow(10, a + ((b - a) * i) / steps));
    }
    return out;
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
    // An explicit selection always wins: only the selected kernels show, and
    // they show regardless of the runtime-threshold filter. This is why
    // isolating a heavy-tail kernel and then lowering the slider never makes
    // the selected kernel vanish.
    if (state.selected.size > 0) {
      return state.selected.has(kernel.index);
    }
    return withinThreshold(kernel);
  }

  // A kernel is actually drawn only if it is visible and has points at the
  // current peak.
  function kernelIsDrawn(kernel) {
    return kernelIsVisible(kernel) && pointsForCurrentPeak(kernel).length > 0;
  }

  function isSoleSelected(kernel) {
    return state.selected.size === 1 && state.selected.has(kernel.index);
  }

  function pointsForCurrentPeak(kernel) {
    var points = kernel.points || [];
    if (state.peak === ALL_PEAKS_VALUE || isSoleSelected(kernel)) {
      return points;
    }
    return points.filter(function (point) {
      return point.peak === state.peak;
    });
  }

  // ===== Compute ceilings / roof isolation =================================

  // Compute ceilings meet the diagonal at the leftmost bandwidth among the isolated roofs.
  function referenceBandwidth() {
    var pool = state.isolatedRoofs.size
      ? rooflineTraces.filter(function (roof) {
          return state.isolatedRoofs.has(roof.traceIndex);
        })
      : rooflineTraces;
    var bws = bandwidthsOf(pool);
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
      if (isolating && refBw && ceilingDenseHi > 0) {
        var left = overlay.peakPerf / refBw;
        var pts = logspace(left, Math.max(ceilingDenseHi, left), ROOF_SAMPLES);
        pts.push(ROOF_EXTREME_MAX_AI);
        xs.push(pts);
        ys.push(
          pts.map(function () {
            return overlay.peakPerf;
          })
        );
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

  // Thicken the roof on the current AI axis.
  // While roofs are being isolated, keep them uniform.
  function applyRoofEmphasis() {
    if (!plotlyReady() || !memoryRoofIndices.length) {
      return;
    }
    var isolating = state.isolatedRoofs.size > 0;
    var emphasizeAll = state.selected.size === 1;
    var widths = rooflineTraces.map(function (roof) {
      if (!isolating && (emphasizeAll || roof.level === state.peak)) {
        return ROOF_WIDTH_EMPHASIS;
      }
      return ROOF_WIDTH_NORMAL;
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

  // ===== Kernel rendering ==================================================

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
      var colorByLevel = isSoleSelected(kernel);
      var baseColor = kernel.color || FALLBACK_COLOR;
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
      markerColors.push(
        points.map(function (point) {
          return colorByLevel ? peakColors[point.peak] || baseColor : baseColor;
        })
      );
      // The full tooltip body is precomputed server-side; just pass it back.
      customdata.push(
        points.map(function (point) {
          return [point.hover];
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

  // ===== Reset view (double-click) =========================================

  // Pad a positive [lo, hi] range in log space and widen it to at least
  // FRAME_MIN_DECADES about its midpoint. Returns the padded range in log10
  // units, ready for a Plotly log-axis range.
  function paddedLogSpan(lo, hi, pad, minDecades) {
    var logLo = Math.log10(lo) - Math.log10(pad);
    var logHi = Math.log10(hi) + Math.log10(pad);
    if (logHi - logLo < minDecades) {
      var mid = 0.5 * (logLo + logHi);
      logLo = mid - 0.5 * minDecades;
      logHi = mid + 0.5 * minDecades;
    }
    return [logLo, logHi];
  }

  // Log-axis frame around the kernel points currently drawn
  // under the active peak, selection, and runtime filter. Returns null when
  // nothing is drawn, so the caller can fall back to the initial view.
  function visibleKernelFrame() {
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
    return {
      x: paddedLogSpan(
        Math.min.apply(null, xs),
        Math.max.apply(null, xs),
        FRAME_PAD,
        FRAME_MIN_DECADES
      ),
      y: paddedLogSpan(
        Math.min.apply(null, ys),
        Math.max.apply(null, ys),
        FRAME_PAD,
        FRAME_MIN_DECADES
      ),
    };
  }

  function resetFrame() {
    var frame = visibleKernelFrame();
    if (!frame && initialRange) {
      frame = { x: initialRange.x.slice(), y: initialRange.y.slice() };
    }
    return frame ? includeRoofSegments(frame, gd.data || []) : null;
  }

  // Double-click handler: re-frame on whatever kernels are currently shown, so
  // reset follows the active filter/selection instead of a fixed spot. With no
  // kernels drawn, restore the baked initial range.
  function resetView() {
    if (!plotlyReady()) {
      return;
    }
    var frame = resetFrame();
    if (frame) {
      Plotly.relayout(gd, { "xaxis.range": frame.x, "yaxis.range": frame.y });
    }
  }

  function clamp(value, minimum, maximum) {
    return Math.min(Math.max(value, minimum), maximum);
  }

  function exportTextWidth(text) {
    if (!exportTextMeasureContext) {
      var canvas = document.createElement("canvas");
      exportTextMeasureContext = canvas.getContext("2d");
      if (exportTextMeasureContext) {
        // Only kernel-legend labels/title are measured here, and the kernel
        // legend is monospace, so measure with the kernel-name font.
        exportTextMeasureContext.font =
          EXPORT_LEGEND_FONT_SIZE + "px " + KERNEL_NAME_FONT_FAMILY;
      }
    }
    if (exportTextMeasureContext) {
      return exportTextMeasureContext.measureText(text).width;
    }
    return (
      text.length *
      EXPORT_LEGEND_FONT_SIZE *
      EXPORT_LEGEND_GLYPH_WIDTH_RATIO
    );
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
    if (kernel.pctRuntime == null || !isFinite(kernel.pctRuntime)) {
      return "";
    }
    return "   " + kernel.pctRuntime.toFixed(2) + "%";
  }

  function exportKernelLabel(kernel, maximumWidth, maximumLines) {
    var runtimeSuffix = kernelExportRuntimeSuffix(kernel);
    return wrapTextToWidth(
      kernel.name,
      maximumWidth,
      maximumLines,
      runtimeSuffix
    );
  }

  function roofLogGeometry(roof, data) {
    var trace = data[roof.traceIndex];
    var xs = (trace && trace.x) || [];
    var bandwidth = Number(roof.bandwidth);
    if (xs.length < 2 || !(xs[0] > 0) || !(xs[xs.length - 1] > 0)) {
      return null;
    }
    if (!(bandwidth > 0)) {
      return null;
    }
    return {
      domainLo: Math.log10(xs[0]),
      domainHi: Math.log10(xs[xs.length - 1]),
      intercept: Math.log10(bandwidth),
    };
  }

  function exportLegendLayout(x, xAnchor, y, yAnchor, fontFamily) {
    return {
      x: x,
      xanchor: xAnchor,
      y: y,
      yanchor: yAnchor,
      bgcolor: "rgba(255,255,255,0.96)",
      bordercolor: "#d7dee8",
      borderwidth: 1,
      font: {
        size: EXPORT_LEGEND_FONT_SIZE,
        family: fontFamily || EXPORT_LEGEND_FONT_FAMILY,
        color: "#1b1f24",
      },
      itemclick: false,
      itemdoubleclick: false,
    };
  }

  function exportKernelLegendTitle(visibleKernelCount) {
    return "Kernels (" + visibleKernelCount + " / " + kernels.length + ")";
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

  function exportKernelLabelLines(kernelCount, plotHeight) {
    if (!kernelCount) {
      return 1;
    }
    var maximumLegendHeight =
      plotHeight * EXPORT_LEGEND_MAX_HEIGHT_RATIO;
    var rowsAvailable = Math.max(
      1,
      Math.floor(
        (maximumLegendHeight - EXPORT_LEGEND_HEADER_HEIGHT) /
          EXPORT_LEGEND_ROW_HEIGHT
      ) - 1
    );
    return clamp(
      Math.floor(rowsAvailable / kernelCount),
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
    var legendHeight =
      EXPORT_LEGEND_HEADER_HEIGHT +
      (visibleKernels.length * kernelLabelLines + 1) *
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
    };
  }

  // Expand symmetrically around the kernel-frame midpoint to expose a segment
  // from every bandwidth roof. Each segment is placed near the kernels rather
  // than at an extrapolated endpoint, so roof visibility cannot shift the
  // kernel cluster away from the centre.
  function includeRoofSegments(frame, data) {
    if (!frame || !frame.x || !frame.y) {
      return frame;
    }

    var originalX = frame.x.slice().sort(function (a, b) {
      return a - b;
    });
    var originalY = frame.y.slice().sort(function (a, b) {
      return a - b;
    });
    var xMid = 0.5 * (originalX[0] + originalX[1]);
    var yMid = 0.5 * (originalY[0] + originalY[1]);
    var xHalfSpan = 0.5 * (originalX[1] - originalX[0]);
    var yHalfSpan = 0.5 * (originalY[1] - originalY[0]);

    rooflineTraces.forEach(function (roof) {
      var geometry = roofLogGeometry(roof, data);
      if (!geometry) {
        return;
      }

      var visibleLength = Math.min(
        FRAME_ROOF_SEGMENT_DECADES,
        geometry.domainHi - geometry.domainLo
      );
      if (!(visibleLength > 0)) {
        return;
      }
      var halfLength = 0.5 * visibleLength;

      var feasibleLo = Math.max(
        geometry.domainLo + halfLength,
        xMid - xHalfSpan + halfLength,
        yMid - yHalfSpan - geometry.intercept + halfLength
      );
      var feasibleHi = Math.min(
        geometry.domainHi - halfLength,
        xMid + xHalfSpan - halfLength,
        yMid + yHalfSpan - geometry.intercept - halfLength
      );
      if (feasibleLo <= feasibleHi) {
        return;
      }

      var segmentMid = 0.5 * (xMid + (yMid - geometry.intercept));
      segmentMid = clamp(
        segmentMid,
        geometry.domainLo + halfLength,
        geometry.domainHi - halfLength
      );
      var segmentXLo = segmentMid - halfLength;
      var segmentXHi = segmentMid + halfLength;
      var segmentYLo = segmentXLo + geometry.intercept;
      var segmentYHi = segmentXHi + geometry.intercept;

      xHalfSpan = Math.max(
        xHalfSpan,
        Math.abs(segmentXLo - xMid),
        Math.abs(segmentXHi - xMid)
      );
      yHalfSpan = Math.max(
        yHalfSpan,
        Math.abs(segmentYLo - yMid),
        Math.abs(segmentYHi - yMid)
      );
    });

    return {
      x: [xMid - xHalfSpan, xMid + xHalfSpan],
      y: [yMid - yHalfSpan, yMid + yHalfSpan],
    };
  }

  // Build an export-only Plotly figure: current chart state on the left and an
  // adaptive static legend on the right. This mirrors Roofline Extractor's PNG
  // layout without serializing the interactive page controls or side panels.
  function buildExportFigure() {
    var data = JSON.parse(JSON.stringify(gd.data));
    var layout = JSON.parse(JSON.stringify(gd.layout));
    var frame = resetFrame();
    if (frame && layout.xaxis && layout.yaxis) {
      layout.xaxis.range = frame.x.slice();
      layout.xaxis.autorange = false;
      layout.yaxis.range = frame.y.slice();
      layout.yaxis.autorange = false;
    }
    data.forEach(function (trace) {
      trace.showlegend = false;
    });

    var visibleKernels = [];
    kernels.forEach(function (kernel, position) {
      var traceIndex = kernelTraceIndices[position];
      if (!kernelIsDrawn(kernel) || !data[traceIndex]) {
        return;
      }
      visibleKernels.push({ kernel: kernel, traceIndex: traceIndex });
    });
    var dimensions = buildExportDimensions(visibleKernels);

    visibleKernels.forEach(function (entry, row) {
      var kernel = entry.kernel;
      var trace = data[entry.traceIndex];
      trace.showlegend = true;
      trace.name = exportKernelLabel(
        kernel,
        dimensions.legendTextWidth,
        dimensions.kernelLabelLines
      );
      trace.legend = "legend";
      trace.legendgroup = "export-kernels";
      trace.legendrank = row;
      if (row === 0) {
        trace.legendgrouptitle = {
          text: exportKernelLegendTitle(visibleKernels.length),
        };
      }
    });

    rooflineTraces.forEach(function (roof, row) {
      var trace = data[roof.traceIndex];
      if (!trace) {
        return;
      }
      trace.showlegend = true;
      trace.name = roof.level;
      trace.legend = "legend2";
      trace.legendgroup = "export-roofs";
      trace.legendrank = EXPORT_ROOF_LEGEND_RANK + row;
      if (row === 0) {
        trace.legendgrouptitle = {
          text: "Bandwidth rooflines (" + rooflineTraces.length + ")",
        };
      }
    });

    layout.autosize = false;
    layout.width = dimensions.width;
    layout.height = dimensions.height;
    layout.showlegend = visibleKernels.length > 0 || rooflineTraces.length > 0;
    layout.hovermode = false;
    layout.dragmode = false;
    layout.margin = layout.margin || {};
    if (dimensions.hasKernelLegend) {
      layout.margin.r =
        (layout.margin.r || 0) + dimensions.legendWidth;
    }
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

  // Rasterize the export-only figure at high resolution. Rendering in a
  // detached off-screen graph keeps the interactive chart's dimensions,
  // selection state, and responsive layout untouched.
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

    // Null-safe so it can run even if setup threw before the graph existed;
    // it always re-enables the button, so the control can never get stuck.
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
            scale: clamp(window.devicePixelRatio || 2, 2, 4),
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

  // ===== Kernel rendering (continued) ======================================

  // Ensure the peak dropdown reflects the current selection.
  function syncPeakControl() {
    if (!peakSelect) {
      return;
    }
    if (state.selected.size === 1) {
      peakSelect.value = ALL_PEAKS_VALUE;
      peakSelect.disabled = true;
      peakSelect.title = "All memory levels are shown for the isolated kernel";
    } else {
      peakSelect.value = state.peak;
      peakSelect.disabled = false;
      peakSelect.title = "";
    }
  }

  function render() {
    syncPeakControl();
    if (!plotlyReady() || !kernelTraceIndices.length) {
      updatePanel();
      updateRoofPanel();
      return;
    }
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
    updatePanel();
    updateRoofPanel();
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
    if (state.selected.size === 1) {
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
    var dataset = opts.dataset || {};
    Object.keys(dataset).forEach(function (key) {
      item.dataset[key] = dataset[key];
    });

    var swatch = document.createElement("span");
    swatch.className = opts.swatchClass || "roofline-swatch";
    swatch.style.backgroundColor = opts.color || FALLBACK_COLOR;

    var label = document.createElement("span");
    label.className = opts.labelClass || "roofline-panel-name";
    label.textContent = opts.label;

    item.appendChild(swatch);
    item.appendChild(label);
    (opts.extras || []).forEach(function (node) {
      item.appendChild(node);
    });
    // Clickable and keyboard-activatable.
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
    (model.peaks || []).forEach(function (peak) {
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
      // Percent of GPU resident time.
      if (kernel.pctRuntime != null && isFinite(kernel.pctRuntime)) {
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
          // Kernel names render monospace.
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
          color: peakColors[roof.level],
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
    // While a single kernel is isolated every level is shown
    // so no single level owns the AI axis -> blank the marker.
    var axisPeak = state.selected.size === 1 ? ALL_PEAKS_VALUE : state.peak;
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
      // Trim rows outside the runtime threshold, but never a selected one.
      item.classList.toggle(
        "filtered",
        !withinThreshold(kernel) && !selected
      );
      // A sole-isolated kernel is drawn across every level (colored by level),
      // so its swatch becomes a gradient of those level colors to match.
      var swatch = item.querySelector(".roofline-swatch");
      if (swatch) {
        if (state.selected.size === 1 && selected) {
          var levelColors = (kernel.points || []).map(function (point) {
            return peakColors[point.peak] || kernel.color || FALLBACK_COLOR;
          });
          swatch.style.background =
            levelColors.length > 1
              ? swatchGradient(levelColors)
              : levelColors[0] || kernel.color || FALLBACK_COLOR;
        } else {
          swatch.style.background = kernel.color || FALLBACK_COLOR;
        }
      }
    });
    // Count how many kernels are actually drawn under the current peak +
    // selection filters, shown as "(drawn / total)" next to the title.
    var shown = 0;
    kernels.forEach(function (kernel) {
      if (kernelIsDrawn(kernel)) {
        shown += 1;
      }
    });
    if (kernelCountEl) {
      kernelCountEl.textContent = formatCount(shown, kernels.length);
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
        var index = Number(runtimeSlider.value);
        state.runtimeThreshold = runtimeBreakpoints[index];
        updateRuntimeLabel();
        render();
      });
    }
    if (resetViewBtn) {
      resetViewBtn.addEventListener("click", resetView);
    }
    if (exportPngBtn) {
      exportPngBtn.addEventListener("click", exportPng);
    }
    if (gd && typeof gd.on === "function") {
      // Plotly owns the chart's pointer interaction layer, so listen through
      // its event emitter instead of relying on a native dblclick bubbling to
      // the outer graph div. Config doubleClick:false suppresses Plotly's
      // static reset while still allowing this event to be observed.
      gd.on("plotly_doubleclick", resetView);
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

  // Plotly's responsive config follows viewport resizes, but the chart's
  // container can also change when the toolbar wraps or the side panel changes
  // size. Observe the actual plot column so its canvas always consumes exactly
  // the remaining space without retaining stale pixel dimensions.
  function schedulePlotResize() {
    if (plotResizeFrame != null) {
      return;
    }
    plotResizeFrame = window.requestAnimationFrame(function () {
      plotResizeFrame = null;
      resizePlot();
    });
  }

  function observePlotContainer() {
    if (plotColumn && typeof window.ResizeObserver === "function") {
      plotResizeObserver = new window.ResizeObserver(schedulePlotResize);
      plotResizeObserver.observe(plotColumn);
      return;
    }
    window.addEventListener("resize", schedulePlotResize);
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
    whenPlotReady(function () {
      captureInitialRange();
      wireEvents();
      observePlotContainer();
      resizePlot();
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
