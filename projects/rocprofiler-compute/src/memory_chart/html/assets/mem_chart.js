// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
(function () {
  "use strict";

  var model = HtmlReport.readModel("mem-chart-model");
  if (!model) {
    return;
  }

  var diagram = document.getElementById("mem-chart-diagram");
  var legend = document.getElementById("mem-chart-legend");
  var tables = document.getElementById("mem-chart-tables");
  var guidance = document.getElementById("mem-chart-guidance");
  var blockElements = new Map();
  var edgeColumns = [];
  var arrowGroups = [];
  var blockColumns = new Map();
  var gridTracks = [];
  var grid;
  var canvas;
  var connectors;
  var scopeHeadingSpace;
  var scopeBoxes = [];
  var camera = { scale: 1, x: 0, y: 0 };
  var fitted = true;
  var pendingLayout = false;
  var selectedLabel = "All kernels";
  var categories = new Set();
  var categoryLabels = {
    read: "Read ←", write: "Write →", atomic: "Atomic ↔",
    util: "Utilization", hit: "Hit rate", stall: "Stall", bw: "Bandwidth"
  };
  var svgNamespace = "http://www.w3.org/2000/svg";

  function element(tag, className, value) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (value != null) {
      node.textContent = String(value);
    }
    return node;
  }

  function categoryClass(category) {
    return Object.prototype.hasOwnProperty.call(categoryLabels, category)
      ? " mem-chart-cat-" + category : "";
  }

  function renderContent(item) {
    categories.add(item.category);
    var row = element("div", "mem-chart-metric" + categoryClass(item.category));
    row.appendChild(element("span", "mem-chart-metric-label", item.title));
    var value = element("span", "mem-chart-value", "N/A");
    value.dataset.slotId = item.slotId;
    row.appendChild(value);
    if (item.bar) {
      var track = element("span", "mem-chart-bar-track");
      var fill = element("span", "mem-chart-bar-fill");
      fill.dataset.barSlot = item.slotId;
      track.appendChild(fill);
      row.appendChild(track);
    }
    return row;
  }

  function renderBlock(block, nested) {
    var node = element("section", "mem-chart-block" + (nested ? " nested" : ""));
    node.dataset.blockId = block.id;
    node.appendChild(element("h3", "mem-chart-block-title", block.title));
    if (block.note) {
      node.appendChild(element("p", "mem-chart-block-note", block.note));
    }
    var arrowMetrics = new Set();
    if (block.position === "above" || block.position === "below") {
      model.layout.arrows.forEach(function (group) {
        if (group.from === block.id || group.to === block.id) {
          group.lanes.forEach(function (lane) { arrowMetrics.add(lane.metric); });
        }
      });
    }
    (block.content || []).filter(function (item) {
      return !arrowMetrics.has(item.metric);
    }).forEach(function (item) {
      node.appendChild(renderContent(item));
    });
    var annotations = element("div", "mem-chart-annotations");
    annotations.dataset.annotationsFor = block.id;
    node.appendChild(annotations);
    if (block.children && block.children.length) {
      var children = element("div", "mem-chart-children");
      block.children.forEach(function (child) {
        children.appendChild(renderBlock(child, true));
      });
      node.appendChild(children);
    }
    blockElements.set(block.id, node);
    return node;
  }

  function renderLane(lane) {
    categories.add(lane.category);
    var row = element("div", "mem-chart-lane" + categoryClass(lane.category));
    row.dataset.direction = lane.direction;
    row.dataset.category = lane.category;
    row.appendChild(element("span", "mem-chart-lane-label", lane.title));
    var value = element("span", "mem-chart-lane-value", "N/A");
    value.dataset.slotId = lane.slotId;
    row.appendChild(value);
    return row;
  }

  function renderArrowGroup(group) {
    var node = element("div", "mem-chart-arrow-group");
    node.dataset.sourceId = group.from;
    node.dataset.targetId = group.to;
    group.lanes.forEach(function (lane) {
      if (lane.groupHeader) {
        node.appendChild(element("div", "mem-chart-lane-group", lane.groupHeader));
      }
      node.appendChild(renderLane(lane));
    });
    arrowGroups.push({ model: group, node: node });
    return node;
  }

  function renderDiagram() {
    var layout = model.layout;
    var columnNumbers = Array.from(new Set(layout.gridBlocks.map(function (block) {
      return block.column;
    }))).sort(function (a, b) { return a - b; });
    var blockColumnNumbers = new Map();
    function registerColumn(block) {
      blockColumnNumbers.set(block.id, block.column);
      block.children.forEach(registerColumn);
    }
    layout.gridBlocks.forEach(registerColumn);
    var columns = new Map();
    var edges = new Map();
    grid = element("div", "mem-chart-grid");
    canvas = element("div", "mem-chart-canvas");
    columnNumbers.forEach(function (column, index) {
      if (index && layout.arrows.some(function (group) {
        return blockColumnNumbers.get(group.to) === column &&
          blockColumnNumbers.get(group.from) !== column;
      })) {
        var edge = element("div", "mem-chart-edge-column");
        edges.set(column, edge);
        edgeColumns.push(edge);
        grid.appendChild(edge);
        gridTracks.push({ edge: edge });
      }
      var blockColumn = element("div", "mem-chart-block-column");
      blockColumn.dataset.column = String(column);
      blockColumn.style.gridColumn = String(gridTracks.length + 1);
      columns.set(column, blockColumn);
      blockColumns.set(column, blockColumn);
      grid.appendChild(blockColumn);
      gridTracks.push({ column: blockColumn });
    });
    grid.style.gridTemplateRows = "auto auto auto auto";
    layout.gridBlocks.forEach(function (block) {
      columns.get(block.column).appendChild(renderBlock(block, false));
    });
    layout.ioBlocks.forEach(function (block) {
      var zone = element("div", "mem-chart-io-zone mem-chart-io-" + block.position);
      zone.dataset.ioId = block.id;
      zone.dataset.hostId = block.host;
      zone.style.gridColumn = columns.get(block.column).style.gridColumn + " / -1";
      zone.style.gridRow = block.position === "below" ? "4" : "1";
      zone.appendChild(renderBlock(block, false));
      grid.appendChild(zone);
    });
    layout.arrows.forEach(function (group) {
      var io = layout.ioBlocks.find(function (block) {
        return block.id === group.to || block.id === group.from;
      });
      var node = renderArrowGroup(group);
      if (io) {
        var connector = element("div", "mem-chart-io-connector");
        connector.appendChild(node);
        blockElements.get(io.id).parentElement.appendChild(connector);
      } else {
        var edge = edges.get(blockColumnNumbers.get(group.to));
        if (edge) {
          edge.appendChild(node);
        }
      }
    });
    connectors = svgElement("svg", { "class": "mem-chart-connectors", "aria-hidden": "true" });
    if (layout.scope) {
      scopeHeadingSpace = element("div", "mem-chart-scope-heading-space");
      scopeHeadingSpace.setAttribute("aria-hidden", "true");
      grid.appendChild(scopeHeadingSpace);
      [layout.scope.gpuLabel, layout.scope.memoryLabel].forEach(function (label, index) {
        var box = element("section", "mem-chart-scope-box" + (index ? " mem-chart-scope-memory" : ""));
        box.setAttribute("aria-label", label);
        box.appendChild(element("h3", "mem-chart-scope-title", label));
        scopeBoxes.push(box);
        canvas.appendChild(box);
      });
    }
    canvas.appendChild(grid);
    canvas.appendChild(connectors);
    diagram.appendChild(canvas);
    renderLegend();
  }

  function svgElement(tag, attributes) {
    var node = document.createElementNS(svgNamespace, tag);
    Object.keys(attributes || {}).forEach(function (key) {
      node.setAttribute(key, attributes[key]);
    });
    return node;
  }

  function rect(node) {
    var origin = canvas.getBoundingClientRect();
    var bounds = node.getBoundingClientRect();
    return {
      left: (bounds.left - origin.left) / camera.scale,
      right: (bounds.right - origin.left) / camera.scale,
      top: (bounds.top - origin.top) / camera.scale,
      bottom: (bounds.bottom - origin.top) / camera.scale
    };
  }

  function sizeBlockColumns() {
    blockElements.forEach(function (block) {
      var empty = !block.querySelector(".mem-chart-metric, .mem-chart-children, .mem-chart-annotation, .mem-chart-block-note");
      block.classList.toggle("mem-chart-block-empty", empty);
    });
    grid.style.gridTemplateColumns = gridTracks.map(function (track) {
      if (track.edge) {
        var width = 140;
        track.edge.querySelectorAll(".mem-chart-lane").forEach(function (lane) {
          var label = lane.querySelector(".mem-chart-lane-label");
          var value = lane.querySelector(".mem-chart-lane-value");
          width = Math.max(width, label.scrollWidth + value.scrollWidth + 11);
        });
        track.edge.querySelectorAll(".mem-chart-lane-group").forEach(function (header) {
          width = Math.max(width, header.scrollWidth + 8);
        });
        return Math.ceil(width) + "px";
      }
      var blocks = Array.from(track.column.children);
      return blocks.every(function (block) {
        return block.classList.contains("mem-chart-block-empty");
      }) ? Math.ceil(Math.max.apply(null, blocks.map(function (block) {
        return block.offsetWidth;
      }))) + "px" : "190px";
    }).join(" ");
    model.layout.ioBlocks.forEach(function (block) {
      var node = blockElements.get(block.id);
      var host = blockElements.get(block.host);
      var zone = node.parentElement;
      node.style.marginLeft = "0px";
      var hostBounds = rect(host);
      var zoneBounds = rect(zone);
      node.style.marginLeft = (hostBounds.left + hostBounds.right) / 2 -
        zoneBounds.left - node.offsetWidth / 2 + "px";
    });
  }

  function reserveLaneHeights() {
    blockElements.forEach(function (block) { block.style.minHeight = ""; });
    edgeColumns.forEach(function (column) { column.style.minHeight = ""; });
    var required = new Map();
    var verticalHeight = 0;
    arrowGroups.forEach(function (group) {
      var edge = group.node.closest(".mem-chart-edge-column");
      if (!edge) {
        verticalHeight = Math.max(verticalHeight, group.node.offsetHeight);
        return;
      }
      [group.model.from, group.model.to].forEach(function (id) {
        var block = blockElements.get(id);
        var edges = required.get(block) || new Map();
        edges.set(edge, (edges.get(edge) || 0) + group.node.offsetHeight + 8);
        required.set(block, edges);
      });
    });
    required.forEach(function (edges, block) {
      var height = Math.max.apply(null, Array.from(edges.values()));
      block.style.minHeight = Math.max(block.offsetHeight, height + 16) + "px";
    });
    var columns = Array.from(grid.querySelectorAll(".mem-chart-block-column"));
    function columnHeight(column) {
      return Array.from(column.children).reduce(function (sum, block) {
        return sum + block.offsetHeight;
      }, Math.max(0, column.children.length - 1) * parseFloat(getComputedStyle(column).rowGap));
    }
    var height = Math.max.apply(null, columns.map(function (column) {
      return columnHeight(column);
    }));
    columns.forEach(function (column) {
      var blocks = Array.from(column.children);
      var natural = columnHeight(column);
      var largest = blocks.reduce(function (a, b) { return a.offsetHeight >= b.offsetHeight ? a : b; });
      largest.style.minHeight = largest.offsetHeight + height - natural + "px";
    });
    // Both attachment sides use the largest bundle, with room for the scope title.
    var gap = parseFloat(getComputedStyle(grid).rowGap);
    var headingHeight = scopeHeadingSpace ? scopeHeadingSpace.offsetHeight : 0;
    var span = verticalHeight + 2 * (headingHeight + gap + 8);
    grid.querySelectorAll(".mem-chart-io-zone").forEach(function (zone) {
      var above = zone.classList.contains("mem-chart-io-above");
      zone.style[above ? "paddingBottom" : "paddingTop"] =
        span - gap - (above && scopeHeadingSpace ? headingHeight + gap : 0) + "px";
    });
  }

  function positionHorizontalGroups() {
    edgeColumns.forEach(function (column) {
      var columnTop = rect(column).top;
      var nextTop = 0;
      arrowGroups.filter(function (group) { return group.node.parentElement === column; })
        .forEach(function (group) {
          var source = rect(blockElements.get(group.model.from));
          var target = rect(blockElements.get(group.model.to));
          var anchor = source.bottom - source.top < target.bottom - target.top ? source : target;
          var center = (anchor.top + anchor.bottom) / 2;
          var top = Math.max(center - columnTop - group.node.offsetHeight / 2, nextTop);
          group.node.style.top = top + "px";
          nextTop = top + group.node.offsetHeight + 8;
        });
      column.style.minHeight = nextTop + "px";
    });
  }

  function verticalBundle(group) {
    var source = rect(blockElements.get(group.model.from));
    var target = rect(blockElements.get(group.model.to));
    var fromIo = blockElements.get(group.model.from).closest(".mem-chart-io-zone");
    var zone = fromIo || blockElements.get(group.model.to).closest(".mem-chart-io-zone");
    var above = zone.classList.contains("mem-chart-io-above");
    var spacing = Math.min(9, Math.max(0, Math.min(source.right - source.left,
      target.right - target.left) - 20) / Math.max(1, group.model.lanes.length - 1));
    return {
      sourceX: (source.left + source.right) / 2,
      targetX: (target.left + target.right) / 2,
      sourceY: fromIo ? (above ? source.bottom : source.top) : (above ? source.top : source.bottom),
      targetY: fromIo ? (above ? target.top : target.bottom) : (above ? target.bottom : target.top),
      spacing: spacing,
      zone: zone
    };
  }

  function positionVerticalGroups() {
    arrowGroups.forEach(function (group) {
      if (!group.node.closest(".mem-chart-io-zone")) { return; }
      var bundle = verticalBundle(group);
      var zone = rect(bundle.zone);
      var connector = group.node.parentElement;
      var halfWidth = (group.model.lanes.length - 1) * bundle.spacing / 2;
      connector.style.left = Math.max(bundle.sourceX, bundle.targetX) + halfWidth + 8 - zone.left + "px";
      connector.style.top = (bundle.sourceY + bundle.targetY) / 2 - zone.top - group.node.offsetHeight / 2 + "px";
    });
  }

  function positionScopeBoxes() {
    if (!scopeHeadingSpace) { return; }
    var columnNumbers = Array.from(blockColumns.keys());
    var splitIndex = columnNumbers.indexOf(model.layout.scope.splitColumn);
    var columns = Array.from(blockColumns.values()).map(rect);
    var split = (columns[splitIndex - 1].right + columns[splitIndex].left) / 2;
    var top = rect(scopeHeadingSpace).top;
    // Include the 1px border so the visible block padding is at least 12px.
    var bottom = Math.max.apply(null, columns.map(function (bounds) { return bounds.bottom; })) + 13;
    var left = columns[0].left - 13;
    var right = columns[columns.length - 1].right + 13;
    scopeBoxes.forEach(function (box, index) {
      var start = index ? split + 2 : left;
      var end = index ? right : split - 2;
      box.style.left = start + "px";
      box.style.top = top + "px";
      box.style.width = end - start + "px";
      box.style.height = bottom - top + "px";
    });
  }

  function alignEdgeGroups() {
    sizeBlockColumns();
    reserveLaneHeights();
    positionHorizontalGroups();
    positionVerticalGroups();
    positionScopeBoxes();
    drawConnectors();
    if (fitted) {
      fitDiagram();
    }
  }

  function drawConnectors() {
    connectors.replaceChildren();
    connectors.setAttribute("width", grid.offsetWidth);
    connectors.setAttribute("height", grid.offsetHeight);
    var defs = svgElement("defs");
    connectors.appendChild(defs);
    var colors = new Map();
    arrowGroups.forEach(function (group) {
      var from = blockElements.get(group.model.from);
      var to = blockElements.get(group.model.to);
      var source = rect(from);
      var target = rect(to);
      var rows = Array.from(group.node.querySelectorAll(".mem-chart-lane"));
      var io = from.closest(".mem-chart-io-zone") || to.closest(".mem-chart-io-zone");
      rows.forEach(function (row, index) {
        var category = row.dataset.category;
        if (!colors.has(category)) {
          var color = getComputedStyle(row).color;
          var marker = svgElement("marker", {
            id: "mem-chart-head-" + category, viewBox: "0 0 8 8", refX: "7", refY: "4",
            markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse",
            markerUnits: "userSpaceOnUse"
          });
          marker.appendChild(svgElement("path", { d: "M 0 0 L 8 4 L 0 8 Z", fill: color }));
          defs.appendChild(marker);
          colors.set(category, color);
        }
        var path;
        if (io) {
          var bundle = verticalBundle(group);
          var offset = (index - (rows.length - 1) / 2) * bundle.spacing;
          var x1 = bundle.sourceX + offset;
          var x2 = bundle.targetX + offset;
          var y1 = bundle.sourceY;
          var y2 = bundle.targetY;
          var middle = (y1 + y2) / 2;
          path = "M " + x1 + " " + y1 + " V " + middle + " H " + x2 + " V " + y2;
        } else {
          var y = rect(row).bottom - 6;
          var yFrom = Math.max(source.top + 10, Math.min(source.bottom - 10, y));
          var yTo = Math.max(target.top + 10, Math.min(target.bottom - 10, y));
          var left = source.right + 6;
          var right = target.left - 6;
          path = "M " + source.right + " " + yFrom + " H " + left + " V " + y +
            " H " + right + " V " + yTo + " H " + target.left;
        }
        var attrs = {
          d: path, "class": "mem-chart-connector" + categoryClass(category),
          "data-from": group.model.from, "data-to": group.model.to,
          stroke: colors.get(category)
        };
        var markerUrl = "url(#mem-chart-head-" + category + ")";
        if (row.dataset.direction !== "forward") {
          attrs["marker-start"] = markerUrl;
        }
        if (row.dataset.direction !== "backward") {
          attrs["marker-end"] = markerUrl;
        }
        connectors.appendChild(svgElement("path", attrs));
      });
    });
    // Label-only attachments still connect to their declared host.
    model.layout.ioBlocks.forEach(function (block) {
      if (model.layout.arrows.some(function (group) { return group.from === block.id || group.to === block.id; })) {
        return;
      }
      var io = rect(blockElements.get(block.id));
      var anchor = blockElements.get(block.host);
      var target = rect(anchor);
      var x = (io.left + io.right) / 2;
      var targetX = (target.left + target.right) / 2;
      var y1 = block.position === "above" ? io.bottom : io.top;
      var y2 = block.position === "above" ? target.top : target.bottom;
      connectors.appendChild(svgElement("path", {
        d: "M " + x + " " + y1 + " V " + (y1 + y2) / 2 + " H " + targetX + " V " + y2,
        "class": "mem-chart-connector", "data-from": block.id, "data-to": block.host
      }));
    });
  }

  function scheduleLayout() {
    if (pendingLayout) { return; }
    pendingLayout = true;
    window.requestAnimationFrame(function () {
      pendingLayout = false;
      alignEdgeGroups();
    });
  }

  function applyCamera() {
    canvas.style.transform = "translate(" + camera.x + "px, " + camera.y + "px) scale(" + camera.scale + ")";
  }

  function fitDiagram() {
    var bounds = renderedBounds();
    var width = bounds.right - bounds.left;
    var height = bounds.bottom - bounds.top;
    camera.scale = Math.max(0.001, Math.min(1, (diagram.clientWidth - 24) / width,
      (diagram.clientHeight - 24) / height));
    camera.x = (diagram.clientWidth - width * camera.scale) / 2 - bounds.left * camera.scale;
    camera.y = (diagram.clientHeight - height * camera.scale) / 2 - bounds.top * camera.scale;
    fitted = true;
    applyCamera();
  }

  function renderedBounds() {
    var bounds = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
    canvas.querySelectorAll(".mem-chart-scope-box, .mem-chart-block, .mem-chart-lane, .mem-chart-lane-label, .mem-chart-lane-value, .mem-chart-lane-group, .mem-chart-connector").forEach(function (node) {
      var box = rect(node);
      // Include connector stroke and arrowheads beyond the SVG path bounds.
      var padding = node.classList.contains("mem-chart-connector") ? 6 : 0;
      bounds.left = Math.min(bounds.left, box.left - padding);
      bounds.right = Math.max(bounds.right, box.right + padding);
      bounds.top = Math.min(bounds.top, box.top - padding);
      bounds.bottom = Math.max(bounds.bottom, box.bottom + padding);
    });
    return bounds;
  }

  function initNavigation() {
    document.getElementById("mem-chart-fit-diagram").addEventListener("click", fitDiagram);
    diagram.addEventListener("dblclick", fitDiagram);
    diagram.addEventListener("wheel", function (event) {
      event.preventDefault();
      var bounds = diagram.getBoundingClientRect();
      var x = event.clientX - bounds.left;
      var y = event.clientY - bounds.top;
      var scale = Math.max(0.08, Math.min(4, camera.scale * Math.exp(-event.deltaY * 0.002)));
      camera.x = x - (x - camera.x) * scale / camera.scale;
      camera.y = y - (y - camera.y) * scale / camera.scale;
      camera.scale = scale;
      fitted = false;
      applyCamera();
    }, { passive: false });
    var drag = null;
    diagram.addEventListener("pointerdown", function (event) {
      if (event.button !== 0) { return; }
      drag = { x: event.clientX, y: event.clientY };
      diagram.setPointerCapture(event.pointerId);
    });
    diagram.addEventListener("pointermove", function (event) {
      if (!drag) { return; }
      camera.x += event.clientX - drag.x;
      camera.y += event.clientY - drag.y;
      drag = { x: event.clientX, y: event.clientY };
      fitted = false;
      applyCamera();
    });
    ["pointerup", "pointercancel", "lostpointercapture"].forEach(function (name) {
      diagram.addEventListener(name, function () { drag = null; });
    });
  }

  function styledClone(node) {
    var clone = node.cloneNode(true);
    var sources = [node].concat(Array.from(node.querySelectorAll("*")));
    var copies = [clone].concat(Array.from(clone.querySelectorAll("*")));
    sources.forEach(function (source, index) {
      var style = getComputedStyle(source);
      for (var property of style) {
        copies[index].style.setProperty(property, style.getPropertyValue(property));
      }
    });
    return clone;
  }

  async function exportPng() {
    var button = document.getElementById("mem-chart-export-png");
    var status = document.getElementById("mem-chart-export-status");
    button.disabled = true;
    status.hidden = true;
    var objectUrl;
    try {
      var width = diagram.clientWidth;
      var heading = document.getElementById("mem-chart-heading").textContent;
      var exportBody = element("div", "");
      exportBody.setAttribute("xmlns", "http://www.w3.org/1999/xhtml");
      exportBody.style.cssText = "padding:12px;box-sizing:border-box;font:13px sans-serif;overflow-wrap:anywhere;";
      exportBody.style.width = width + 24 + "px";
      exportBody.style.background = getComputedStyle(diagram).backgroundColor;
      exportBody.style.color = getComputedStyle(document.body).color;
      var normalization = document.getElementById("mem-chart-normalization").textContent;
      var title = element("h2", "", heading);
      if (normalization) {
        title.appendChild(element("strong", "", " " + normalization));
      }
      exportBody.appendChild(title);
      exportBody.appendChild(element("p", "", selectedLabel + document.getElementById("mem-chart-scope").textContent));
      var snapshot = styledClone(diagram);
      snapshot.style.width = width + "px";
      snapshot.style.height = diagram.clientHeight + "px";
      snapshot.style.cursor = "default";
      snapshot.removeAttribute("tabindex");
      exportBody.appendChild(snapshot);
      exportBody.appendChild(styledClone(legend));
      exportBody.style.position = "fixed";
      exportBody.style.left = "-100000px";
      document.body.appendChild(exportBody);
      var height = exportBody.offsetHeight;
      exportBody.remove();
      exportBody.style.position = "static";
      exportBody.style.left = "auto";
      var svg = svgElement("svg", { xmlns: svgNamespace, width: width + 24, height: height });
      var foreign = svgElement("foreignObject", { width: "100%", height: "100%" });
      foreign.appendChild(exportBody);
      svg.appendChild(foreign);
      // A data URL keeps the SVG origin-clean for canvas export with foreignObject.
      objectUrl = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(svg));
      var picture = new Image();
      await new Promise(function (resolve, reject) {
        picture.onload = resolve;
        picture.onerror = reject;
        picture.src = objectUrl;
      });
      var raster = document.createElement("canvas");
      var scale = Math.min(2, 4096 / Math.max(width + 24, height));
      raster.width = Math.ceil((width + 24) * scale);
      raster.height = Math.ceil(height * scale);
      var context = raster.getContext("2d");
      context.scale(scale, scale);
      context.drawImage(picture, 0, 0);
      var blob = await new Promise(function (resolve) { raster.toBlob(resolve, "image/png"); });
      if (!blob) { throw new Error("No PNG image was produced"); }
      objectUrl = URL.createObjectURL(blob);
      var link = element("a", "");
      link.download = "mem_chart.png";
      link.href = objectUrl;
      link.click();
    } catch (error) {
      status.textContent = "PNG export failed. Please try again.";
      status.hidden = false;
      console.error("Memory chart PNG export failed", error);
    } finally {
      if (objectUrl && objectUrl.startsWith("blob:")) {
        setTimeout(function () { URL.revokeObjectURL(objectUrl); }, 1000);
      }
      button.disabled = false;
    }
  }

  function renderLegend(view) {
    var present = new Set(categories);
    if (view && view.membw && view.membw.annotations.length) {
      present.add("stall");
    }
    legend.replaceChildren();
    Object.keys(categoryLabels).forEach(function (category) {
      if (!present.has(category)) {
        return;
      }
      var item = element("span", "mem-chart-legend-item" + categoryClass(category));
      item.appendChild(element("span", "mem-chart-legend-mark", "●"));
      item.appendChild(element("span", "", categoryLabels[category]));
      legend.appendChild(item);
    });
  }

  function updateSlots(view) {
    diagram.querySelectorAll("[data-slot-id]").forEach(function (node) {
      var slot = view.slots[node.dataset.slotId];
      node.textContent = slot ? slot.text + (slot.unitLabel ? " " + slot.unitLabel : "")
        : "N/A";
    });
    diagram.querySelectorAll("[data-bar-slot]").forEach(function (node) {
      var slot = view.slots[node.dataset.barSlot];
      node.style.width = slot && slot.percent != null ? slot.percent + "%" : "0%";
      node.parentElement.classList.toggle("unavailable", !slot || slot.percent == null);
    });
  }

  function updateAnnotations(view) {
    diagram.querySelectorAll("[data-annotations-for]").forEach(function (node) {
      node.replaceChildren();
    });
    if (!view.membw) {
      return;
    }
    view.membw.annotations.forEach(function (annotation) {
      var block = blockElements.get(annotation.blockId);
      if (!block) {
        return;
      }
      var target = block.querySelector("[data-annotations-for]");
      var row = element("div", "mem-chart-annotation");
      row.appendChild(element("span", "", annotation.label));
      row.appendChild(element("strong", "", annotation.value));
      target.appendChild(row);
    });
  }

  function updateTables(view) {
    tables.replaceChildren();
    if (!view.tables.length) {
      tables.appendChild(element("p", "mem-chart-empty", "No metric tables are available."));
      return;
    }
    view.tables.forEach(function (table) {
      var section = element("section", "mem-chart-table-wrap");
      section.appendChild(element("h3", "", table.title));
      var scroll = element("div", "mem-chart-table-scroll");
      var tableNode = element("table", "mem-chart-table");
      var colgroup = element("colgroup", "");
      var shares = table.columns.length + (table.columns.includes("Metric") ? 1 : 0);
      table.columns.forEach(function (column) {
        var col = element("col", "");
        col.style.width = (100 * (column === "Metric" ? 2 : 1) / shares) + "%";
        colgroup.appendChild(col);
      });
      tableNode.appendChild(colgroup);
      var head = element("thead", "");
      var headerRow = element("tr", "");
      table.columns.forEach(function (column) {
        headerRow.appendChild(element("th", "", column));
      });
      head.appendChild(headerRow);
      tableNode.appendChild(head);
      var body = element("tbody", "");
      table.rows.forEach(function (values) {
        var row = element("tr", "");
        values.forEach(function (value) {
          row.appendChild(element("td", "", value == null ? "N/A" : value));
        });
        body.appendChild(row);
      });
      tableNode.appendChild(body);
      scroll.appendChild(tableNode);
      section.appendChild(scroll);
      tables.appendChild(section);
    });
  }

  function updateGuidance(view) {
    guidance.replaceChildren();
    var detail = view.membw;
    guidance.hidden = !detail;
    if (!detail) {
      return;
    }
    guidance.appendChild(element("h2", "", "Memory bandwidth guidance"));
    if (detail.status) {
      guidance.appendChild(element("p", "mem-chart-status", detail.status));
    }
    detail.guidanceBlocks.forEach(function (block) {
      guidance.appendChild(element("p", "mem-chart-guidance-block", block));
    });
  }

  function selectView(key) {
    var kernel = model.kernels.find(function (item) { return item.index === key; });
    var view = kernel ? kernel.view : model.aggregate;
    selectedLabel = kernel ? kernel.name : "All kernels";
    document.getElementById("mem-chart-view-label").textContent = selectedLabel;
    document.getElementById("mem-chart-selection").textContent = selectedLabel;
    document.title = model.heading + " — " + selectedLabel;
    updateSlots(view);
    updateAnnotations(view);
    updateTables(view);
    updateGuidance(view);
    renderLegend(view);
    scheduleLayout();
  }

  document.getElementById("mem-chart-heading").textContent = model.heading;
  var scope = model.scopeNote.replace(/^All kernels(?: · )?/, "");
  document.getElementById("mem-chart-scope").textContent = scope ? " · " + scope : "";
  var normalization = model.normalization ? model.normalization.trim() : "";
  document.getElementById("mem-chart-normalization").textContent = normalization
    ? "(Normalization: " + normalization.replace(/_/g, " ") + ")" : "";
  renderDiagram();
  var kernelList = HtmlReport.createKernelList({
    list: document.getElementById("mem-chart-kernel-list"),
    countElement: document.getElementById("mem-chart-kernel-count"),
    showAllButton: document.getElementById("mem-chart-show-all"),
    mode: "single",
    items: model.kernels.map(function (kernel) {
      return { key: kernel.index, label: kernel.name, color: "var(--report-accent)" };
    }),
    onChange: function (selected) {
      selectView(selected.size ? Array.from(selected)[0] : null);
    }
  });
  kernelList.setCountText("(" + model.kernels.length + ")");
  if (model.initialKernel != null) {
    kernelList.select(model.initialKernel);
  } else {
    selectView(null);
  }
  HtmlReport.initTheme(document.getElementById("mem-chart-theme-toggle"), scheduleLayout);
  initNavigation();
  var sidebarToggle = document.getElementById("mem-chart-sidebar-toggle");
  var kernelPanel = document.getElementById("mem-chart-kernel-panel");
  sidebarToggle.addEventListener("click", function () {
    var collapsed = !kernelPanel.hidden;
    kernelPanel.hidden = collapsed;
    kernelPanel.parentElement.classList.toggle("collapsed", collapsed);
    sidebarToggle.setAttribute("aria-expanded", String(!collapsed));
    var label = (collapsed ? "Expand" : "Collapse") + " kernel sidebar";
    sidebarToggle.setAttribute("aria-label", label);
    sidebarToggle.title = label;
    sidebarToggle.textContent = collapsed ? "‹" : "›";
    scheduleLayout();
  });
  document.getElementById("mem-chart-export-png").addEventListener("click", exportPng);
  new ResizeObserver(scheduleLayout).observe(diagram);
  if (document.fonts) { document.fonts.ready.then(scheduleLayout); }
  window.addEventListener("resize", scheduleLayout);
})();
