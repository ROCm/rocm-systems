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
  var categories = new Set();
  var categoryLabels = {
    read: "Read ←", write: "Write →", atomic: "Atomic ↔",
    util: "Utilization", hit: "Hit rate", stall: "Stall", bw: "Bandwidth"
  };
  var directionGlyphs = { backward: "←", forward: "→", both: "↔" };

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
    (block.content || []).forEach(function (item) {
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
    row.appendChild(element("span", "mem-chart-lane-arrow",
      directionGlyphs[lane.direction] || "↔"));
    row.appendChild(element("span", "mem-chart-lane-label", lane.title));
    var value = element("span", "mem-chart-lane-value", "N/A");
    value.dataset.slotId = lane.slotId;
    row.appendChild(value);
    return row;
  }

  function renderArrowGroup(group) {
    var node = element("div", "mem-chart-arrow-group");
    node.dataset.targetId = group.to;
    group.lanes.forEach(function (lane) {
      if (lane.groupHeader) {
        node.appendChild(element("div", "mem-chart-lane-group", lane.groupHeader));
      }
      node.appendChild(renderLane(lane));
    });
    return node;
  }

  function renderDiagram() {
    var layout = model.layout;
    var allBlocks = layout.gridBlocks.concat(layout.ioBlocks);
    var maxColumn = Math.max.apply(null, allBlocks.map(function (block) {
      return block.column;
    }));
    var grid = element("div", "mem-chart-grid");
    var columns = [];
    var widths = [];
    for (var column = 0; column <= maxColumn; column += 1) {
      var blockColumn = element("div", "mem-chart-block-column");
      blockColumn.dataset.column = String(column);
      var above = element("div", "mem-chart-io-zone mem-chart-io-above");
      var stack = element("div", "mem-chart-block-stack");
      var below = element("div", "mem-chart-io-zone mem-chart-io-below");
      blockColumn.appendChild(above);
      blockColumn.appendChild(stack);
      blockColumn.appendChild(below);
      columns.push({ above: above, stack: stack, below: below });
      grid.appendChild(blockColumn);
      widths.push("190px");
      if (column < maxColumn) {
        var edge = element("div", "mem-chart-edge-column");
        edgeColumns.push(edge);
        grid.appendChild(edge);
        widths.push("145px");
      }
    }
    grid.style.gridTemplateColumns = widths.join(" ");
    layout.gridBlocks.forEach(function (block) {
      columns[block.column].stack.appendChild(renderBlock(block, false));
    });
    layout.ioBlocks.forEach(function (block) {
      var zone = block.position === "below" ? "below" : "above";
      columns[block.column][zone].appendChild(renderBlock(block, false));
    });
    layout.arrows.forEach(function (group) {
      if (layout.ioBlocks.some(function (block) { return block.id === group.to; })) {
        var ioBlock = blockElements.get(group.to);
        var connector = element("div", "mem-chart-io-connector");
        connector.appendChild(renderArrowGroup(group));
        ioBlock.appendChild(connector);
      } else {
        var targetColumn = Number(blockElements.get(group.to).closest(
          ".mem-chart-block-column").dataset.column);
        var index = Math.max(0, Math.min(edgeColumns.length - 1, targetColumn - 1));
        edgeColumns[index].appendChild(renderArrowGroup(group));
      }
    });
    diagram.appendChild(grid);
    renderLegend();
  }

  function alignEdgeGroups() {
    edgeColumns.forEach(function (column) {
      var columnTop = column.getBoundingClientRect().top;
      var nextTop = 0;
      Array.from(column.children).forEach(function (group) {
        var target = blockElements.get(group.dataset.targetId);
        var targetTop = target.getBoundingClientRect().top - columnTop;
        var top = Math.max(targetTop, nextTop);
        group.style.top = top + "px";
        nextTop = top + group.offsetHeight + 8;
      });
      column.style.minHeight = Math.max(260, nextTop) + "px";
    });
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
    document.getElementById("mem-chart-view-label").textContent = kernel
      ? kernel.name : "All kernels";
    updateSlots(view);
    updateAnnotations(view);
    updateTables(view);
    updateGuidance(view);
    renderLegend(view);
    window.requestAnimationFrame(alignEdgeGroups);
  }

  document.getElementById("mem-chart-heading").textContent = model.heading;
  document.getElementById("mem-chart-scope").textContent = model.scopeNote;
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
  HtmlReport.initTheme(document.getElementById("mem-chart-theme-toggle"), function () {});
  window.addEventListener("resize", alignEdgeGroups);
})();
