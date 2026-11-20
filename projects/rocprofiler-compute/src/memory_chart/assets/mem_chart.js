// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Draws the memory chart from its layout file (src/memory_chart/README.md) and
// fills it with the values of one view: the aggregate, or one kernel.
(function () {
  "use strict";

  var model = window.HtmlReport.readModel("mem-chart-model");
  var clamp = window.HtmlReport.clamp;
  if (!model) {
    return;
  }

  var SVG_NS = "http://www.w3.org/2000/svg";
  var LANE_GAP = 8;
  var layout = model.layout;
  var diagram = document.getElementById("mem-chart-diagram");
  var legend = document.getElementById("mem-chart-legend");
  var guidance = document.getElementById("mem-chart-guidance");
  var blocksById = {};
  var nodesById = {};
  var columnNodes = [];
  var edgeGroups = [];
  var verticals = [];
  var canvas;
  var grid;
  var connectors;
  var camera = { scale: 1, x: 0, y: 0 };
  var fitted = true;
  var view = model.views.aggregate;

  layout.blocks.forEach(function (block) { blocksById[block.id] = block; });
  var columnCount = 1 + Math.max.apply(null, layout.blocks.map(function (block) {
    return block.column;
  }));

  // ---- Building the chart from the layout ---------------------------------

  function element(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text != null) {
      node.textContent = String(text);
    }
    return node;
  }

  function svgElement(tag, attributes) {
    var node = document.createElementNS(SVG_NS, tag);
    Object.keys(attributes).forEach(function (key) {
      node.setAttribute(key, attributes[key]);
    });
    return node;
  }

  // A nested block has no row of its own; it is in its parent's row.
  function rowOf(block) {
    return block.parent ? rowOf(blocksById[block.parent]) : block.row;
  }

  function byOrder(a, b) {
    return a.order - b.order;
  }

  function childrenOf(block) {
    return layout.blocks.filter(function (other) {
      return other.parent === block.id;
    }).sort(byOrder);
  }

  function valueNode(metricId) {
    var node = element("span", "mem-chart-value", "N/A");
    node.dataset.metric = metricId;
    return node;
  }

  // Hovering a value shows its metric's name and description from the panel config
  function describe(node, metricId) {
    var metric = model.metrics[metricId];
    if (metric) {
      node.title = metric.description ? metric.name + ": " + metric.description : metric.name;
    }
  }

  function renderItem(item) {
    var row = element("div", "mem-chart-item mem-cat-" + item.category);
    describe(row, item.metric_id);
    row.appendChild(element("span", "mem-chart-item-title", item.title));
    row.appendChild(valueNode(item.metric_id));
    if (item.bar) {
      var bar = barNode(null);
      bar.firstChild.dataset.bar = item.metric_id;
      row.appendChild(bar);
    }
    return row;
  }

  // A bar for a percentage, in the category color
  function barNode(percent) {
    var bar = element("span", "mem-chart-bar");
    var fill = element("span", "");
    fillBar(fill, percent);
    bar.appendChild(fill);
    return bar;
  }

  function fillBar(fill, percent) {
    fill.style.width = clamp(percent || 0, 0, 100) + "%";
  }

  function renderBlock(block) {
    var node = element(
      "section",
      "mem-chart-block mem-chart-" + block.style
    );
    node.appendChild(element("h3", "mem-chart-block-title", block.title));
    (block.content || []).forEach(function (item) {
      node.appendChild(renderItem(item));
    });
    childrenOf(block).forEach(function (child) {
      node.appendChild(renderBlock(child));
    });
    if (block.stall_level) {
      var stalls = element("div", "mem-chart-stalls");
      stalls.dataset.stallsFor = block.id;
      node.appendChild(stalls);
    }
    nodesById[block.id] = node;
    return node;
  }

  function renderLanes(arrows) {
    var lanes = element("div", "mem-chart-lanes");
    var group = null;
    arrows.forEach(function (arrow) {
      if (arrow.group && arrow.group !== group) {
        lanes.appendChild(element("div", "mem-chart-lane-group", arrow.group));
      }
      group = arrow.group;
      var lane = element(
        "div",
        "mem-chart-lane mem-chart-lane-" + arrow.style + " mem-cat-" + arrow.category
      );
      lane.appendChild(element("span", "mem-chart-lane-label", arrow.title));
      describe(lane, arrow.metric_id);
      lane.appendChild(valueNode(arrow.metric_id));
      lane.arrow = arrow;
      lanes.appendChild(lane);
    });
    return lanes;
  }

  function topLevel(column) {
    return layout.blocks.filter(function (block) {
      return block.row === 0 && !block.parent && block.column === column;
    }).sort(byOrder);
  }

  function place(node, column, row) {
    node.style.gridColumn = String(2 * column + 1);
    node.style.gridRow = String(row);
  }

  function renderScopes() {
    var split = blocksById[layout.scope.split].column;
    layout.scope.labels.forEach(function (label, index) {
      var box = element("div", "mem-chart-scope mem-chart-scope-" + index, label);
      box.style.gridColumn = index ? 2 * split + 1 + " / -1" : "1 / " + 2 * split;
      box.style.gridRow = "3";
      grid.appendChild(box);
    });
  }

  function renderEdges() {
    var tracks = [];
    for (var column = 0; column < columnCount; column++) {
      tracks.push("max-content");
      if (column === columnCount - 1) {
        break;
      }
      var edge = element("div", "mem-chart-edge");
      edge.style.gridColumn = String(2 * column + 2);
      edge.style.gridRow = "3";
      // The arrows grouped as the terminal chart groups them (render.edge_segments)
      var segments = model.edges[column];
      segments.forEach(function (segment) {
        var arrows = segment.arrows.map(function (index) { return layout.arrows[index]; });
        var node = renderLanes(arrows);
        edge.appendChild(node);
        edgeGroups.push({ anchor: segment.anchor, node: node, edge: edge });
      });
      edge.classList.toggle("mem-chart-edge-empty", !segments.length);
      grid.appendChild(edge);
      tracks.push("max-content");
    }
    grid.style.gridTemplateColumns = tracks.join(" ");
  }

  function renderAttached(block, above) {
    var row = above ? 1 : 5;
    var holder = element(
      "div",
      "mem-chart-attached mem-chart-attached-" + (above ? "above" : "below")
    );
    holder.appendChild(renderBlock(block));
    place(holder, block.column, row);
    grid.appendChild(holder);
    var arrows = layout.arrows.filter(function (arrow) {
      return arrow.to === block.id || arrow.from === block.id;
    });
    var host = topLevel(block.column);
    var vertical = {
      block: block,
      above: above,
      host: above ? host[0] : host[host.length - 1],
      lanes: null,
    };
    var lanesHolder = element(
      "div",
      "mem-chart-vertical mem-chart-vertical-" + (above ? "above" : "below")
    );
    if (arrows.length) {
      vertical.lanes = renderLanes(arrows);
      lanesHolder.appendChild(vertical.lanes);
    }
    place(lanesHolder, block.column, above ? 2 : 4);
    grid.appendChild(lanesHolder);
    verticals.push(vertical);
  }

  function renderChart() {
    canvas = element("div", "mem-chart-canvas");
    grid = element("div", "mem-chart-grid");
    renderScopes();
    for (var column = 0; column < columnCount; column++) {
      var node = element("div", "mem-chart-column");
      topLevel(column).forEach(function (block) {
        node.appendChild(renderBlock(block));
      });
      place(node, column, 3);
      grid.appendChild(node);
      columnNodes.push(node);
    }
    renderEdges();
    layout.blocks.forEach(function (block) {
      if (rowOf(block) !== 0) {
        renderAttached(block, block.row < 0);
      }
    });
    connectors = svgElement("svg", { "class": "mem-chart-connectors", "aria-hidden": "true" });
    canvas.appendChild(grid);
    canvas.appendChild(connectors);
    diagram.appendChild(canvas);
  }

  // ---- Values of the selected view ----------------------------------------

  function showView() {
    diagram.querySelectorAll("[data-metric]").forEach(function (node) {
      var value = view.values[node.dataset.metric];
      node.textContent = value ? value.text : "N/A";
    });
    diagram.querySelectorAll("[data-bar]").forEach(function (node) {
      var value = view.values[node.dataset.bar];
      fillBar(node, value ? value.number : null);
    });
    diagram.querySelectorAll("[data-stalls-for]").forEach(function (node) {
      var rows = view.stalls[node.dataset.stallsFor] || [];
      node.replaceChildren();
      rows.forEach(function (stall) {
        // A stall with guidance links to it, below the chart
        var linked = stall.guidance != null;
        var row = element(
          linked ? "a" : "div",
          "mem-chart-item mem-chart-stall mem-cat-stall"
        );
        if (linked) {
          row.href = "#mem-chart-guidance-" + stall.guidance;
          row.title = "Show the guidance for this stall";
        }
        row.appendChild(element("span", "mem-chart-item-title", "⚠︎ " + stall.label));
        row.appendChild(element("span", "mem-chart-value", stall.text));
        row.appendChild(barNode(stall.number));
        node.appendChild(row);
      });
      node.closest(".mem-chart-block").classList.toggle("mem-chart-stalled", rows.length > 0);
    });
    showLegend();
    showGuidance();
    scheduleLayout();
  }

  function showLegend() {
    var used = {};
    layout.blocks.forEach(function (block) {
      (block.content || []).forEach(function (item) { used[item.category] = true; });
    });
    layout.arrows.forEach(function (arrow) { used[arrow.category] = true; });
    used.stall = Object.keys(view.stalls).some(function (id) {
      return view.stalls[id].length > 0;
    });
    legend.replaceChildren();
    model.legend.forEach(function (entry) {
      if (used[entry.category]) {
        legend.appendChild(element("span", "mem-cat-" + entry.category, entry.label));
      }
    });
  }

  function showGuidance() {
    guidance.replaceChildren();
    if (view.status) {
      guidance.appendChild(element("p", "mem-chart-status", view.status));
    }
    view.guidance.forEach(function (text, index) {
      var block = element("pre", "mem-chart-guidance-block");
      block.id = "mem-chart-guidance-" + index;
      var lines = text.split("\n");
      block.appendChild(element("strong", "", lines[0]));
      block.appendChild(document.createTextNode("\n" + lines.slice(1).join("\n")));
      guidance.appendChild(block);
    });
  }

  // ---- Geometry that depends on rendered sizes ----------------------------

  function rect(node) {
    var origin = grid.getBoundingClientRect();
    var bounds = node.getBoundingClientRect();
    return {
      left: (bounds.left - origin.left) / camera.scale,
      right: (bounds.right - origin.left) / camera.scale,
      top: (bounds.top - origin.top) / camera.scale,
      bottom: (bounds.bottom - origin.top) / camera.scale,
    };
  }

  // Panel blocks share the width of the widest one, so block sizes say nothing
  // about the hardware.
  function evenPanelWidths() {
    var panels = Array.from(grid.querySelectorAll(".mem-chart-column > .mem-chart-panel"));
    canvas.style.removeProperty("--mem-panel-width");
    panels.forEach(function (panel) { panel.style.width = "max-content"; });
    var width = Math.max.apply(null, panels.map(function (panel) {
      // offsetWidth rounds down, which would wrap the widest row
      var bounds = rect(panel);
      return Math.ceil(bounds.right - bounds.left);
    }).concat([0]));
    panels.forEach(function (panel) { panel.style.width = ""; });
    canvas.style.setProperty("--mem-panel-width", width + "px");
  }

  // The terminal chart's height rules (render.MemChart._heights), in pixels:
  // a block is at least as tall as the arrows that line up with it, on either
  // side, and those arrows start at its top. Arrows between two single blocks
  // sit in the middle of the edge. So every arrow runs straight.
  function placeEdgeGroups() {
    grid.querySelectorAll(".mem-chart-edge").forEach(function (edge) {
      // The lanes stretch to the edge, so measure them at the edge's own width
      edge.style.width = "";
      var width = 0;
      edge.querySelectorAll(".mem-chart-lanes").forEach(function (lanes) {
        width = Math.max(width, lanes.scrollWidth);
      });
      edge.style.width = width ? width + 2 * LANE_GAP + 16 + "px" : "";
    });
    Object.keys(nodesById).forEach(function (id) { nodesById[id].style.minHeight = ""; });
    var needs = {};
    edgeGroups.forEach(function (group) {
      if (group.anchor) {
        needs[group.anchor] = Math.max(
          needs[group.anchor] || 0,
          group.node.offsetHeight + 2 * LANE_GAP
        );
      }
    });
    Object.keys(needs).forEach(function (id) {
      nodesById[id].style.minHeight = needs[id] + "px";
    });
    edgeGroups.forEach(function (group) {
      var edge = rect(group.edge);
      var top = group.anchor
        ? rect(nodesById[group.anchor]).top - edge.top + LANE_GAP
        : (edge.bottom - edge.top - group.node.offsetHeight) / 2;
      group.node.style.top = top + "px";
    });
  }

  function marker(category, color) {
    var id = "mem-chart-head-" + category;
    if (!connectors.querySelector("#" + id)) {
      var head = svgElement("marker", {
        id: id, viewBox: "0 0 8 8", refX: "7", refY: "4", markerWidth: "7",
        markerHeight: "7", orient: "auto-start-reverse", markerUnits: "userSpaceOnUse",
      });
      head.appendChild(svgElement("path", { d: "M 0 0 L 8 4 L 0 8 Z", fill: color }));
      connectors.firstChild.appendChild(head);
    }
    return "url(#" + id + ")";
  }

  function connector(d, lane) {
    var path = svgElement("path", { d: d, "class": "mem-cat-" + lane.arrow.category });
    var head = marker(
      lane.arrow.category,
      getComputedStyle(lane).getPropertyValue("--mem-category").trim()
    );
    // d runs from the arrow's "from" block to its "to" block; "forward" points to "to"
    if (lane.arrow.direction !== "forward") {
      path.setAttribute("marker-start", head);
    }
    if (lane.arrow.direction !== "backward") {
      path.setAttribute("marker-end", head);
    }
    connectors.appendChild(path);
  }

  function drawConnectors() {
    connectors.replaceChildren(svgElement("defs", {}));
    connectors.setAttribute("width", grid.offsetWidth);
    connectors.setAttribute("height", grid.offsetHeight);
    edgeGroups.forEach(function (group) {
      group.node.querySelectorAll(".mem-chart-lane").forEach(function (lane) {
        var source = rect(nodesById[lane.arrow.from]);
        var target = rect(nodesById[lane.arrow.to]);
        var y = rect(lane).bottom + 1;
        // Kept off the block corners, inside the margin placeEdgeGroups leaves
        var y1 = clamp(y, source.top + LANE_GAP / 2, source.bottom - LANE_GAP / 2);
        var y2 = clamp(y, target.top + LANE_GAP / 2, target.bottom - LANE_GAP / 2);
        connector(
          "M " + source.right + " " + y1 + " H " + (source.right + LANE_GAP) +
            " V " + y + " H " + (target.left - LANE_GAP) + " V " + y2 +
            " H " + target.left,
          lane
        );
      });
    });
    verticals.forEach(function (vertical) {
      var block = rect(nodesById[vertical.block.id]);
      var host = rect(nodesById[vertical.host.id]);
      var x = (host.left + host.right) / 2;
      var y1 = vertical.above ? host.top : host.bottom;
      var y2 = vertical.above ? block.bottom : block.top;
      var lanes = vertical.lanes ? Array.from(vertical.lanes.querySelectorAll(".mem-chart-lane")) : [];
      if (!lanes.length) {
        connectors.appendChild(svgElement("path", { d: "M " + x + " " + y1 + " V " + y2 }));
        return;
      }
      lanes.forEach(function (lane, index) {
        var offset = (index - (lanes.length - 1) / 2) * 6;
        // A vertical arrow starts at its column's block and ends at the attached one
        var fromHost = lane.arrow.from === vertical.host.id;
        connector(
          fromHost
            ? "M " + (x + offset) + " " + y1 + " V " + y2
            : "M " + (x + offset) + " " + y2 + " V " + y1,
          lane
        );
      });
    });
  }

  var layoutPending = false;

  function scheduleLayout() {
    if (layoutPending) {
      return;
    }
    layoutPending = true;
    window.requestAnimationFrame(function () {
      layoutPending = false;
      evenPanelWidths();
      placeEdgeGroups();
      drawConnectors();
      if (fitted) {
        fit();
      }
    });
  }

  // ---- Navigation and export ----------------------------------------------

  function applyCamera() {
    canvas.style.transform =
      "translate(" + camera.x + "px, " + camera.y + "px) scale(" + camera.scale + ")";
  }

  function fit() {
    var width = canvas.offsetWidth;
    var height = canvas.offsetHeight;
    camera.scale = Math.min(1, diagram.clientWidth / width, diagram.clientHeight / height);
    camera.x = (diagram.clientWidth - width * camera.scale) / 2;
    camera.y = (diagram.clientHeight - height * camera.scale) / 2;
    fitted = true;
    applyCamera();
  }

  function initNavigation() {
    document.getElementById("mem-chart-fit").addEventListener("click", fit);
    diagram.addEventListener("dblclick", fit);
    diagram.addEventListener("wheel", function (event) {
      if (!event.ctrlKey && !event.metaKey) {
        return;
      }
      event.preventDefault();
      var bounds = diagram.getBoundingClientRect();
      var x = event.clientX - bounds.left;
      var y = event.clientY - bounds.top;
      var scale = clamp(camera.scale * Math.exp(-event.deltaY * 0.002), 0.08, 4);
      camera.x = x - (x - camera.x) * scale / camera.scale;
      camera.y = y - (y - camera.y) * scale / camera.scale;
      camera.scale = scale;
      fitted = false;
      applyCamera();
    }, { passive: false });
    var drag = null;
    diagram.addEventListener("pointerdown", function (event) {
      if (event.button === 0 && !event.target.closest("a")) {
        drag = { x: event.clientX, y: event.clientY };
        diagram.setPointerCapture(event.pointerId);
      }
    });
    diagram.addEventListener("pointermove", function (event) {
      if (!drag) {
        return;
      }
      camera.x += event.clientX - drag.x;
      camera.y += event.clientY - drag.y;
      drag = { x: event.clientX, y: event.clientY };
      fitted = false;
      applyCamera();
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach(function (name) {
      diagram.addEventListener(name, function () { drag = null; });
    });
    window.addEventListener("resize", function () {
      if (fitted) {
        fit();
      }
    });
  }

  // Copies computed styles inline, so the chart draws the same inside an image.
  function styledClone(node) {
    var clone = node.cloneNode(true);
    var sources = [node].concat(Array.from(node.querySelectorAll("*")));
    var copies = [clone].concat(Array.from(clone.querySelectorAll("*")));
    sources.forEach(function (source, index) {
      var style = getComputedStyle(source);
      for (var i = 0; i < style.length; i++) {
        copies[index].style.setProperty(style[i], style.getPropertyValue(style[i]));
      }
    });
    return clone;
  }

  function exportPng() {
    var button = document.getElementById("mem-chart-export-png");
    button.disabled = true;
    var saved = { scale: camera.scale, x: camera.x, y: camera.y };
    camera = { scale: 1, x: 0, y: 0 };
    applyCamera();
    var width = canvas.offsetWidth;
    var height = canvas.offsetHeight + legend.offsetHeight + 8;
    var page = element("div", "");
    page.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
    page.style.background = getComputedStyle(diagram).backgroundColor;
    var copy = styledClone(canvas);
    copy.style.position = "static";
    copy.style.transform = "none";
    page.appendChild(copy);
    page.appendChild(styledClone(legend));
    camera = saved;
    applyCamera();
    var svg = svgElement("svg", { xmlns: SVG_NS, width: width, height: height });
    var foreign = svgElement("foreignObject", { width: "100%", height: "100%" });
    foreign.appendChild(page);
    svg.appendChild(foreign);
    var picture = new Image();
    picture.onload = function () {
      var scale = Math.min(2, 8192 / Math.max(width, height));
      var raster = element("canvas", "");
      raster.width = Math.ceil(width * scale);
      raster.height = Math.ceil(height * scale);
      var context = raster.getContext("2d");
      context.scale(scale, scale);
      context.drawImage(picture, 0, 0);
      raster.toBlob(function (blob) {
        var link = element("a", "");
        link.download = document.title.replace(/[^\w.-]+/g, "_") + ".png";
        link.href = URL.createObjectURL(blob);
        link.click();
        setTimeout(function () { URL.revokeObjectURL(link.href); }, 1000);
        button.disabled = false;
      }, "image/png");
    };
    picture.onerror = function () {
      button.disabled = false;
    };
    // A data URL keeps the canvas origin-clean, so it can be saved
    picture.src = "data:image/svg+xml;charset=utf-8," +
      encodeURIComponent(new XMLSerializer().serializeToString(svg));
  }

  // ---- Start ---------------------------------------------------------------

  document.getElementById("mem-chart-heading").textContent = model.heading;
  renderChart();
  initNavigation();
  document.getElementById("mem-chart-export-png").addEventListener("click", exportPng);
  window.HtmlReport.initTheme(
    document.getElementById("mem-chart-theme-toggle"),
    scheduleLayout
  );
  showView();
})();
