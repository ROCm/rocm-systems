// Painted text and CSS padding must share the rendered coordinate space.
export async function readMetricGeometry(page) {
  await page.evaluate(() => document.fonts.ready);
  const strip = page.getByTestId('comparison-metric-strip');
  await strip.scrollIntoViewIfNeeded();
  return strip.evaluate((element) => {
    const rect = (r) => ({ left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height });
    const textRects = (node) => {
      const walker = document.createTreeWalker(node, NodeFilter.SHOW_TEXT);
      const lines = [];
      while (walker.nextNode()) {
        const text = walker.currentNode;
        if (!text.textContent.trim() || text.parentElement.closest('svg')) continue;
        const range = document.createRange();
        range.selectNodeContents(text);
        lines.push(...Array.from(range.getClientRects(), rect));
      }
      return lines;
    };
    return Array.from(element.children).map((cell) => {
      const style = getComputedStyle(cell);
      const box = cell.getBoundingClientRect();
      let scale = 1;
      for (let ancestor = cell; ancestor; ancestor = ancestor.parentElement) scale *= parseFloat(getComputedStyle(ancestor).zoom) || 1;
      const typography = Array.from(cell.children).filter((child) => child.classList.contains('MuiTypography-root'));
      const [label, value, caption, ...details] = typography;
      // These auto-width normal-flow blocks span the cell content box. Use its
      // actually laid-out edges, not a width ratio rounded by CSS serialization.
      const flow = caption.getBoundingClientRect();
      const content = {
        left: flow.left,
        right: flow.right,
        top: label.getBoundingClientRect().top,
        bottom: box.bottom - (parseFloat(style.paddingBottom) + parseFloat(style.borderBottomWidth)) * scale,
      };
      return { id: cell.dataset.testid, content, scale, text: value.textContent, value: textRects(value),
        label: textRects(label), after: [caption, ...details].flatMap(textRects),
        allText: textRects(cell), valueStyle: { overflow: getComputedStyle(value).overflow, textOverflow: getComputedStyle(value).textOverflow },
      };
    });
  });
}

export function metricContainmentFailures(cells) {
  const failures = [];
  const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
  for (const cell of cells) {
    if (!cell.value.length) failures.push(`${cell.id}: missing painted value`);
    if (cell.valueStyle.overflow !== 'visible' || cell.valueStyle.textOverflow === 'ellipsis') failures.push(`${cell.id}: clipped value`);
    for (const line of cell.value) {
      for (const edge of ['left', 'top']) if (line[edge] < cell.content[edge]) failures.push(`${cell.id}: value outside ${edge}`);
      for (const edge of ['right', 'bottom']) if (line[edge] > cell.content[edge]) failures.push(`${cell.id}: value outside ${edge}`);
      if ([...cell.label, ...cell.after].some((r) => overlaps(line, r))) failures.push(`${cell.id}: caption/commit collision`);
      if (cells.some((other) => other !== cell && other.allText.some((r) => overlaps(line, r)))) failures.push(`${cell.id}: adjacent text collision`);
    }
  }
  return failures;
}
