# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Executable browser-controller tests using Node's built-in VM module."""

import shutil
import subprocess
from pathlib import Path

import pytest

_CONTROLLER = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "roofline"
    / "assets"
    / "roofline_plot.js"
)

_NODE_HARNESS = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const source = fs.readFileSync(process.argv[2], "utf8");

class ClassList {
  constructor(owner) {
    this.owner = owner;
  }

  values() {
    return this.owner.className.split(/\s+/).filter(Boolean);
  }

  contains(name) {
    return this.values().includes(name);
  }

  toggle(name, force) {
    const names = new Set(this.values());
    const enabled = force === undefined ? !names.has(name) : force;
    if (enabled) {
      names.add(name);
    } else {
      names.delete(name);
    }
    this.owner.className = Array.from(names).join(" ");
    return enabled;
  }
}

function eventOf(type, values = {}) {
  return Object.assign(
    {
      type,
      target: null,
      defaultPrevented: false,
      propagationStopped: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      stopPropagation() {
        this.propagationStopped = true;
      },
    },
    values
  );
}

class Element {
  constructor(tagName, id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.className = "";
    this.classList = new ClassList(this);
    this.children = [];
    this.childNodes = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {
      setProperty() {},
    };
    this.listeners = {};
    this.hidden = false;
    this.disabled = false;
    this.textContent = "";
    this.title = "";
    this.attributes = {};
  }

  appendChild(child) {
    child.parentNode = this;
    this.childNodes.push(child);
    if (child.tagName !== "#TEXT") {
      this.children.push(child);
    }
    return child;
  }

  addEventListener(type, listener) {
    this.listeners[type] = this.listeners[type] || [];
    this.listeners[type].push(listener);
  }

  dispatchEvent(event) {
    if (!event.target) {
      event.target = this;
    }
    event.currentTarget = this;
    (this.listeners[event.type] || []).forEach((listener) => listener(event));
    if (!event.propagationStopped && this.parentNode) {
      this.parentNode.dispatchEvent(event);
    }
    if (
      event.type === "keydown" &&
      this.tagName === "BUTTON" &&
      (event.key === "Enter" || event.key === " ") &&
      !event.defaultPrevented
    ) {
      this.dispatchEvent(eventOf("click"));
    }
    return !event.defaultPrevented;
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] || null;
  }

  querySelector(selector) {
    const className = selector.startsWith(".") ? selector.slice(1) : null;
    for (const child of this.children) {
      if (className && child.classList.contains(className)) {
        return child;
      }
      const nested = child.querySelector(selector);
      if (nested) {
        return nested;
      }
    }
    return null;
  }

  closest(selector) {
    return selector === ".roofline-plot-col" ? this.plotColumn || null : null;
  }

  scrollIntoView() {}

  remove() {
    if (!this.parentNode) {
      return;
    }
    this.parentNode.children = this.parentNode.children.filter(
      (child) => child !== this
    );
    this.parentNode.childNodes = this.parentNode.childNodes.filter(
      (child) => child !== this
    );
    this.parentNode = null;
  }

  getContext() {
    return {
      font: "",
      measureText(text) {
        return { width: text.length * 7 };
      },
    };
  }
}

function immediateThenable() {
  return {
    then(resolve, reject) {
      try {
        const result = resolve ? resolve() : undefined;
        return result && typeof result.then === "function"
          ? result
          : immediateThenable();
      } catch (error) {
        if (reject) {
          reject(error);
          return immediateThenable();
        }
        throw error;
      }
    },
  };
}

function deferredThenable() {
  let resolveCallback;
  let rejectCallback;
  return {
    promise: {
      then(resolve, reject) {
        resolveCallback = resolve;
        rejectCallback = reject;
      },
    },
    resolve() {
      resolveCallback();
    },
    reject() {
      rejectCallback();
    },
  };
}

function baseModel() {
  return {
    allPeaksValue: "__all__",
    roofExtremeMaxAi: 1e6,
    kernelNameFontFamily: "monospace",
    divId: "plot",
    frame: { x: [1e-2, 1e2], y: [1, 1e4] },
    kernels: [],
    kernelTraceIndices: [],
    rooflineTraces: [],
    computeTraces: [],
    computeOverlayTraces: [],
    precisions: ["FP32"],
    peakColors: { HBM: "#123456" },
    peaks: ["HBM"],
    defaultPeak: "HBM",
  };
}

function makeController(overrides = {}) {
  const model = Object.assign(baseModel(), overrides);
  const elements = {};
  const plotEvents = {};
  const documentEvents = {};
  const animationFrames = [];
  const resizeObservers = [];
  const modelElement = new Element("script", "roofline-model");
  modelElement.textContent = JSON.stringify(model);
  const graph = new Element("div", "plot");
  const plotColumn = new Element("div");
  graph.plotColumn = plotColumn;
  graph.clientWidth = 400;
  graph.clientHeight = 400;
  graph.layout = {
    margin: { l: 0, r: 0, t: 0, b: 0 },
    xaxis: { range: [-2, 2] },
    yaxis: { range: [0, 4] },
    title: { text: "Roofline" },
  };
  const traceIndices = model.kernelTraceIndices
    .concat(model.rooflineTraces.map((roof) => roof.traceIndex))
    .concat(model.computeTraces.map((trace) => trace.traceIndex))
    .concat(model.computeOverlayTraces.map((trace) => trace.traceIndex));
  const traceCount = traceIndices.length ? Math.max(...traceIndices) + 1 : 0;
  graph.data = Array.from({ length: traceCount }, () => ({
    type: "scatter",
    mode: "markers",
    name: "trace",
    x: [1],
    y: [1],
    marker: { color: "#123456", line: {} },
    customdata: [[]],
  }));
  model.kernels.forEach((kernel, index) => {
    const traceIndex = model.kernelTraceIndices[index];
    if (traceIndex == null) {
      return;
    }
    graph.data[traceIndex].name = kernel.name;
    graph.data[traceIndex].x = kernel.points.map((point) => point.ai);
    graph.data[traceIndex].y = kernel.points.map((point) => point.perf);
  });
  model.rooflineTraces.forEach((roof) => {
    graph.data[roof.traceIndex].mode = "lines";
    graph.data[roof.traceIndex].name = roof.level;
    graph.data[roof.traceIndex].x = roof.x || [1e-2, roof.kneeAi || 1];
    graph.data[roof.traceIndex].y = roof.y || [1, roof.kneePerf || 1e3];
  });
  model.computeTraces.forEach((trace) => {
    graph.data[trace.traceIndex].mode = "lines";
    graph.data[trace.traceIndex].name = trace.label || trace.dtype;
  });
  graph.on = (name, listener) => {
    plotEvents[name] = plotEvents[name] || [];
    plotEvents[name].push(listener);
  };
  elements["roofline-model"] = modelElement;
  elements.plot = graph;
  const tags = {
    "roofline-precision-btn": "button",
    "roofline-precision-menu": "div",
    "roofline-precision-label": "span",
    "roofline-peak-select": "select",
    "roofline-peak-control": "label",
    "roofline-kernel-list": "ul",
    "roofline-show-all": "button",
    "roofline-kernel-count": "span",
    "roofline-kernel-offplot-count": "span",
    "roofline-runtime-threshold": "input",
    "roofline-runtime-value": "span",
    "roofline-runtime-filter": "div",
    "roofline-roof-list": "ul",
    "roofline-roof-count": "span",
    "roofline-show-all-roofs": "button",
    "roofline-reset-view": "button",
    "roofline-fit-data": "button",
    "roofline-export-png": "button",
    "roofline-theme-toggle": "button",
  };
  Object.keys(tags).forEach((id) => {
    elements[id] = new Element(tags[id], id);
  });
  elements["roofline-precision-menu"].hidden = true;
  elements["roofline-export-png"].textContent = "Export PNG";

  const relayoutCalls = [];
  const restyleCalls = [];
  const relayoutResults = [];
  const newPlotCalls = [];
  const downloadCalls = [];
  let purgeCalls = 0;
  const plotly = {
    relayout(_graph, update) {
      relayoutCalls.push(update);
      if (update["xaxis.range"]) {
        graph.layout.xaxis.range = update["xaxis.range"].slice();
      }
      if (update["yaxis.range"]) {
        graph.layout.yaxis.range = update["yaxis.range"].slice();
      }
      return update["xaxis.range"] && relayoutResults.length
        ? relayoutResults.shift()
        : immediateThenable();
    },
    restyle(_graph, update, indices) {
      restyleCalls.push({ update, indices });
      return immediateThenable();
    },
    newPlot(target, data, layout, config) {
      target.data = data;
      target.layout = layout;
      newPlotCalls.push({ target, data, layout, config });
      return immediateThenable();
    },
    downloadImage(target, options) {
      downloadCalls.push({ target, options });
      return immediateThenable();
    },
    purge() {
      purgeCalls += 1;
    },
    Plots: {
      resize() {},
    },
  };
  const document = {
    readyState: "loading",
    title: "Roofline",
    documentElement: new Element("html"),
    body: new Element("body"),
    getElementById(id) {
      return elements[id] || null;
    },
    createElement(tagName) {
      return new Element(tagName);
    },
    createTextNode(text) {
      const node = new Element("#text");
      node.textContent = text;
      return node;
    },
    addEventListener(type, listener) {
      documentEvents[type] = documentEvents[type] || [];
      documentEvents[type].push(listener);
    },
    emit(type, event = eventOf(type)) {
      (documentEvents[type] || []).forEach((listener) => listener(event));
    },
  };
  const requestedHooks = {};
  const window = {
    __rooflineTestHooks: requestedHooks,
    requestAnimationFrame(callback) {
      animationFrames.push(callback);
      return animationFrames.length;
    },
    ResizeObserver: class {
      constructor(callback) {
        this.callback = callback;
        resizeObservers.push(this);
      }

      observe(target) {
        this.target = target;
      }
    },
    matchMedia() {
      return {
        matches: false,
        addEventListener() {},
      };
    },
  };
  const context = vm.createContext({
    console,
    document,
    isFinite,
    JSON,
    Math,
    Plotly: plotly,
    Set,
    setTimeout,
    window,
    getComputedStyle() {
      return {
        getPropertyValue() {
          return "";
        },
      };
    },
  });
  vm.runInContext(source, context, { filename: "roofline_plot.js" });
  document.readyState = "complete";
  document.emit("DOMContentLoaded");
  return {
    document,
    downloadCalls,
    elements,
    emitPlot(name, event) {
      (plotEvents[name] || []).forEach((listener) => listener(event));
    },
    flushAnimationFrames() {
      while (animationFrames.length) {
        animationFrames.shift()();
      }
    },
    graph,
    hooks: requestedHooks,
    newPlotCalls,
    plotEvents,
    purgeCalls() {
      return purgeCalls;
    },
    relayoutCalls,
    relayoutResults,
    resizeObservers,
    restyleCalls,
    triggerResize() {
      resizeObservers.forEach((observer) => observer.callback());
    },
  };
}

function rangeCalls(controller) {
  return controller.relayoutCalls.filter((update) => update["xaxis.range"]);
}

{
  const controller = makeController({
    kernels: [
      {
        name: "inside",
        color: "#123456",
        points: [{ ai: 1, perf: 10, peak: "HBM", hoverCells: [] }],
      },
    ],
    kernelTraceIndices: [0],
  });
  const initialRanges = rangeCalls(controller);
  assert.equal(initialRanges.length, 1);
  assert.deepEqual(initialRanges[0]["xaxis.range"], [-2, 2]);
  assert.deepEqual(initialRanges[0]["yaxis.range"], [0, 4]);
  assert.equal(controller.resizeObservers.length, 1);
  [
    "plotly_relayout",
    "plotly_doubleclick",
    "plotly_click",
  ].forEach((name) => {
    assert.equal(controller.plotEvents[name].length, 1);
  });
}

{
  const controller = makeController();
  const older = deferredThenable();
  const newer = deferredThenable();
  controller.relayoutResults.push(older.promise, newer.promise);
  controller.elements["roofline-reset-view"].dispatchEvent(eventOf("click"));
  controller.elements["roofline-reset-view"].dispatchEvent(eventOf("click"));
  assert.equal(controller.hooks.isApplyingRange(), true);
  older.resolve();
  assert.equal(
    controller.hooks.isApplyingRange(),
    true,
    "a stale resolution released the current range operation"
  );
  newer.resolve();
  assert.equal(controller.hooks.isApplyingRange(), false);
}

{
  const controller = makeController({
    kernels: [
      {
        name: "inside",
        color: "#123456",
        points: [{ ai: 10, perf: 100, peak: "HBM", hoverCells: [] }],
      },
    ],
    kernelTraceIndices: [0],
  });
  controller.graph.layout.xaxis.range = [8, 9];
  controller.graph.layout.yaxis.range = [10, 11];
  controller.emitPlot("plotly_relayout", { "xaxis.range[0]": 8 });
  const manualCalls = rangeCalls(controller).length;
  controller.triggerResize();
  controller.flushAnimationFrames();
  assert.equal(
    rangeCalls(controller).length,
    manualCalls,
    "resize replaced a manual range"
  );

  controller.elements["roofline-fit-data"].dispatchEvent(eventOf("click"));
  const fittedCalls = rangeCalls(controller).length;
  assert.equal(controller.hooks.isAutoFramed(), false);
  controller.triggerResize();
  controller.flushAnimationFrames();
  assert.equal(
    rangeCalls(controller).length,
    fittedCalls,
    "resize replaced a one-shot fit range"
  );
}

{
  const malformed = [
    { ai: 2, perf: null, peak: "HBM" },
    { ai: null, perf: 3, peak: "HBM" },
    { ai: -1, perf: 4, peak: "HBM" },
  ];
  const controller = makeController({
    kernels: [{ name: "invalid", color: "#123456", points: malformed }],
    kernelTraceIndices: [0],
  });
  const kernel = controller.hooks.kernels()[0];
  const rangesBeforeFit = rangeCalls(controller).length;
  controller.elements["roofline-fit-data"].dispatchEvent(eventOf("click"));
  const row = controller.elements["roofline-kernel-list"].children[0];
  const badge = row.querySelector(".roofline-kernel-offplot");
  assert.equal(controller.hooks.kernelIsDrawn(kernel), false);
  assert.equal(controller.hooks.drawnPoints().length, 0);
  const payload = controller.hooks.buildKernelRestylePayload();
  assert.equal(payload.xs[0].length, 0);
  assert.equal(payload.ys[0].length, 0);
  assert.equal(payload.visibility[0], false);
  assert.equal(rangeCalls(controller).length, rangesBeforeFit);
  assert.equal(controller.elements["roofline-fit-data"].disabled, true);
  assert.equal(
    controller.elements["roofline-kernel-offplot-count"].hidden,
    true
  );
  assert.equal(badge.hidden, true);
}

{
  const controller = makeController({
    precisions: ["FP32", "FP64"],
    computeTraces: [
      { traceIndex: 0, dtype: "FP32", peakPerf: 100, label: "FP32" },
      { traceIndex: 1, dtype: "FP64", peakPerf: 80, label: "FP64" },
    ],
  });
  const rangesBeforePrecision = rangeCalls(controller).length;
  const xBefore = controller.graph.layout.xaxis.range.slice();
  const yBefore = controller.graph.layout.yaxis.range.slice();
  const precisionButton = controller.elements["roofline-precision-btn"];
  const precisionMenu = controller.elements["roofline-precision-menu"];
  precisionButton.dispatchEvent(eventOf("click"));
  const fp64Label = precisionMenu.children.find((label) =>
    label.children.some((child) => child.value === "FP64")
  );
  assert.ok(fp64Label, "expected an FP64 precision option");
  const fp64 = fp64Label.children.find((child) => child.value === "FP64");
  const restylesBeforePrecision = controller.restyleCalls.length;
  fp64.checked = true;
  precisionMenu.dispatchEvent(eventOf("change", { target: fp64 }));
  assert.equal(precisionMenu.hidden, false);
  assert.equal(rangeCalls(controller).length, rangesBeforePrecision);
  assert.equal(controller.restyleCalls.length, restylesBeforePrecision + 1);
  assert.deepEqual(controller.graph.layout.xaxis.range, xBefore);
  assert.deepEqual(controller.graph.layout.yaxis.range, yBefore);
  controller.document.emit("click");
  assert.equal(precisionMenu.hidden, true);
}

{
  const controller = makeController({
    precisions: ["FP32", "FP64"],
    computeTraces: [
      { traceIndex: 0, dtype: "FP32", peakPerf: 100, label: "FP32" },
      { traceIndex: 1, dtype: "FP64", peakPerf: 40, label: "FP64" },
    ],
    rooflineTraces: [
      {
        traceIndex: 2,
        level: "HBM",
        bandwidth: 10,
        kneeAi: 10,
        kneePerf: 100,
        x: [0.1, 8, 2, 10],
        y: [1, 80, 20, 100],
      },
    ],
  });
  const precisionMenu = controller.elements["roofline-precision-menu"];
  const precisionInput = (name) =>
    precisionMenu.children
      .flatMap((label) => label.children)
      .find((child) => child.value === name);
  const fp32 = precisionInput("FP32");
  const fp64 = precisionInput("FP64");
  const roofUpdates = () =>
    controller.restyleCalls.filter(
      (call) => call.indices.includes(2) && call.update.x
    );

  fp64.checked = true;
  precisionMenu.dispatchEvent(eventOf("change", { target: fp64 }));
  fp32.checked = false;
  precisionMenu.dispatchEvent(eventOf("change", { target: fp32 }));
  const lowUpdates = roofUpdates();
  const low = lowUpdates[lowUpdates.length - 1].update;
  const lowX = Array.from(low.x[0]);
  const lowY = Array.from(low.y[0]);
  assert.deepEqual(lowX, [0.1, 2, 4]);
  assert.deepEqual(lowY, [1, 20, 40]);
  assert.ok(lowY.every((value) => value <= 40));
  assert.deepEqual([lowX[lowX.length - 1], lowY[lowY.length - 1]], [4, 40]);
  assert.ok(lowX.every((value, index) => index === 0 || value > lowX[index - 1]));
  assert.ok(lowY.every((value, index) => index === 0 || value > lowY[index - 1]));

  fp32.checked = true;
  precisionMenu.dispatchEvent(eventOf("change", { target: fp32 }));
  fp64.checked = false;
  precisionMenu.dispatchEvent(eventOf("change", { target: fp64 }));
  const restoredUpdates = roofUpdates();
  const restored = restoredUpdates[restoredUpdates.length - 1].update;
  assert.deepEqual(Array.from(restored.x[0]), [0.1, 2, 8, 10]);
  assert.deepEqual(Array.from(restored.y[0]), [1, 20, 80, 100]);
}

{
  const controller = makeController({
    kernels: [
      {
        name: "far kernel",
        color: "#123456",
        pctRuntime: 100,
        points: [
          { ai: 1e3, perf: 10, peak: "HBM", hoverCells: [] },
          { ai: 1e-2, perf: 1e3, peak: "L2", hoverCells: [] },
        ],
      },
    ],
    kernelTraceIndices: [0],
    peaks: ["HBM", "L2"],
  });
  const row = controller.elements["roofline-kernel-list"].children[0];
  const action = row.querySelector(".roofline-panel-action");
  const label = row.querySelector(".roofline-kernel-name");
  const percentage = row.querySelector(".roofline-kernel-pct");
  const badge = row.querySelector(".roofline-kernel-offplot");
  assert.equal(row.getAttribute("role"), null);
  assert.equal(action.tagName, "BUTTON");
  assert.equal(action.parentNode, row);
  assert.equal(label.parentNode, action);
  assert.equal(percentage.parentNode, action);
  assert.equal(badge.parentNode, row);
  assert.equal(badge.hidden, false);
  assert.equal(
    badge.getAttribute("aria-label"),
    "Zoom to off-plot kernel far kernel"
  );
  label.dispatchEvent(eventOf("click"));
  assert.equal(row.classList.contains("selected"), true);
  percentage.dispatchEvent(eventOf("click"));
  assert.equal(row.classList.contains("selected"), false);
  action.dispatchEvent(eventOf("keydown", { key: "Enter" }));
  assert.equal(row.classList.contains("selected"), true);
  action.dispatchEvent(eventOf("click"));
  assert.equal(row.classList.contains("selected"), false);
  const rangesBeforeZoom = rangeCalls(controller).length;
  const click = eventOf("click");
  badge.dispatchEvent(click);
  assert.equal(click.propagationStopped, true);
  assert.equal(row.classList.contains("selected"), false);
  assert.equal(rangeCalls(controller).length, rangesBeforeZoom + 1);
  const badgeZoomCalls = rangeCalls(controller);
  const badgeZoom = badgeZoomCalls[badgeZoomCalls.length - 1];
  assert.deepEqual(Array.from(badgeZoom["xaxis.range"]), [-2.5, 3.5]);
  assert.deepEqual(Array.from(badgeZoom["yaxis.range"]), [-1, 5]);
  badge.dispatchEvent(eventOf("keydown", { key: "Enter" }));
  assert.equal(row.classList.contains("selected"), false);
  assert.equal(rangeCalls(controller).length, rangesBeforeZoom + 2);
}

{
  const controller = makeController({
    frame: { x: [1e-2, 1e2], y: [1, 1e6] },
    kernels: [
      {
        name: "tall frame kernel",
        color: "#123456",
        points: [{ ai: 10, perf: 100, peak: "HBM", hoverCells: [] }],
      },
    ],
    kernelTraceIndices: [0],
  });
  controller.elements["roofline-fit-data"].dispatchEvent(eventOf("click"));
  const zoomCalls = rangeCalls(controller);
  const zoom = zoomCalls[zoomCalls.length - 1];
  const zoomX = Array.from(zoom["xaxis.range"]);
  const zoomY = Array.from(zoom["yaxis.range"]);
  assert.deepEqual(zoomX, [0.5, 1.5]);
  assert.deepEqual(zoomY, [1.25, 2.75]);
  assert.ok(zoomX[0] <= 1 && zoomX[1] >= 1, "zoom cropped the kernel intensity");
  assert.ok(zoomY[0] <= 2 && zoomY[1] >= 2, "zoom cropped the kernel performance");
  const frameAspect = (2 - -2) / (6 - 0);
  assert.ok(
    Math.abs((zoomX[1] - zoomX[0]) / (zoomY[1] - zoomY[0]) - frameAspect) < 1e-12,
    "zoom changed the decades-per-axis ratio a roof knee is drawn with"
  );
}

{
  const controller = makeController({
    rooflineTraces: [
      {
        traceIndex: 0,
        level: "HBM",
        bandwidth: 100,
        kneeAi: 10,
        kneePerf: 1000,
      },
    ],
  });
  const row = controller.elements["roofline-roof-list"].children[0];
  const action = row.querySelector(".roofline-panel-action");
  const axisLabel = row.querySelector(".roofline-roof-aiaxis");
  assert.equal(axisLabel.parentNode, action);
  axisLabel.dispatchEvent(eventOf("click"));
  assert.equal(row.classList.contains("selected"), true);
}

{
  const controller = makeController();
  controller.graph.layout.xaxis.range = [8, 9];
  controller.graph.layout.yaxis.range = [10, 11];
  controller.emitPlot("plotly_doubleclick");
  const ranges = rangeCalls(controller);
  const reset = ranges[ranges.length - 1];
  assert.deepEqual(reset["xaxis.range"], [-2, 2]);
  assert.deepEqual(reset["yaxis.range"], [0, 4]);
}

{
  const controller = makeController({
    kernels: [
      {
        name: "exported",
        color: "#123456",
        points: [{ ai: 10, perf: 100, peak: "HBM", hoverCells: [] }],
      },
    ],
    kernelTraceIndices: [0],
  });
  controller.elements["roofline-fit-data"].dispatchEvent(eventOf("click"));
  const currentX = Array.from(controller.graph.layout.xaxis.range);
  const currentY = Array.from(controller.graph.layout.yaxis.range);
  controller.elements["roofline-export-png"].dispatchEvent(eventOf("click"));
  assert.equal(controller.newPlotCalls.length, 1);
  assert.deepEqual(
    Array.from(controller.newPlotCalls[0].layout.xaxis.range),
    currentX
  );
  assert.deepEqual(
    Array.from(controller.newPlotCalls[0].layout.yaxis.range),
    currentY
  );
  assert.equal(controller.downloadCalls.length, 1);
  assert.equal(controller.purgeCalls(), 1);
}
"""


def test_roofline_browser_controller_behaviors(tmp_path: Path) -> None:
    """Exercise viewport and panel behavior in a dependency-free Node harness."""
    assert _CONTROLLER.is_file(), f"roofline controller not found: {_CONTROLLER}"
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for roofline browser-controller tests")

    runner = tmp_path / "roofline_plot_runner.js"
    runner.write_text(_NODE_HARNESS, encoding="utf-8")
    completed = subprocess.run(
        [node, str(runner), str(_CONTROLLER)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
