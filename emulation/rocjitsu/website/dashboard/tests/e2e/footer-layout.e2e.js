import { expect, test } from '@playwright/test';
import { installPublication, openDashboard, ready } from './helpers/dashboard.js';

async function measureFooter(page) {
  return page.evaluate(() => {
    const footer = document.querySelector('footer');
    const container = document.querySelector('main > .MuiContainer-root');
    const bounds = container.getBoundingClientRect();
    const style = getComputedStyle(container);
    const scale = bounds.width / container.offsetWidth;
    const scroll = document.scrollingElement;
    return {
      footer: footer.getBoundingClientRect().toJSON(),
      paragraphs: [...footer.querySelectorAll('p')].map((p) => ({ text: p.textContent, bounds: p.getBoundingClientRect().toJSON() })),
      content: { left: bounds.left + (parseFloat(style.borderLeftWidth) + parseFloat(style.paddingLeft)) * scale,
        right: bounds.right - (parseFloat(style.borderRightWidth) + parseFloat(style.paddingRight)) * scale },
      viewport: { width: innerWidth, height: innerHeight },
      scroll: { top: scroll.scrollTop, max: scroll.scrollHeight - scroll.clientHeight, height: scroll.scrollHeight, width: scroll.scrollWidth },
    };
  });
}

function assertParagraphs(geometry) {
  expect(geometry.paragraphs.map(({ text }) => text)).toEqual([
    'Copyright © 2025–2026 Advanced Micro Devices, Inc.', 'MIT License',
  ]);
}

function assertTextInside(geometry, paragraph, index) {
  const label = `paragraph ${index} (${paragraph.text})`;
  expect(paragraph.bounds.top, `${label} text top`).toBeGreaterThan(0);
  expect(paragraph.bounds.left, `${label} text left`).toBeGreaterThan(0);
  expect(paragraph.bounds.right, `${label} text right`).toBeLessThan(geometry.viewport.width);
  expect(paragraph.bounds.bottom, `${label} text bottom`).toBeLessThan(geometry.viewport.height);
}

function assertFooterFrame(geometry) {
  expect(geometry.scroll.top, 'actual reachable maximum scroll').toBe(geometry.scroll.max);
  expect(geometry.scroll.width, 'exact no horizontal overflow').toBe(geometry.viewport.width);
  expect(geometry.footer.top, 'footer top').toBeGreaterThanOrEqual(0);
  expect(geometry.footer.left, 'footer left').toBeGreaterThanOrEqual(0);
  expect(geometry.footer.right, 'footer right').toBeLessThanOrEqual(geometry.viewport.width);
}

function assertBottom(geometry) {
  expect(geometry.footer.bottom - geometry.viewport.height, 'max-scroll bottom residual').toBeLessThan(1);
}

function assertVisible(geometry) {
  assertParagraphs(geometry);
  assertFooterFrame(geometry);
  assertBottom(geometry);
  geometry.paragraphs.forEach((p, index) => assertTextInside(geometry, p, index));
}

function assertAligned(geometry) {
  assertParagraphs(geometry);
  for (const [index, paragraph] of geometry.paragraphs.entries()) {
    expect(Math.abs(paragraph.bounds.left - geometry.content.left), `paragraph ${index} left alignment`).toBeLessThan(1);
    expect(Math.abs(paragraph.bounds.right - geometry.content.right), `paragraph ${index} right alignment`).toBeLessThan(1);
  }
  expect(geometry.scroll.width).toBe(geometry.viewport.width);
}

const attachGeometry = (name, value) => test.info().attach(name, { body: JSON.stringify(value, null, 2), contentType: 'application/json' });
const restoreStyle = (node, style) => node.evaluate((element, original) => original === null ? element.removeAttribute('style') : element.setAttribute('style', original), style);

for (const viewport of [{ width: 1440, height: 900 }, { width: 3840, height: 2160 }, { width: 390, height: 1800 }]) {
  test(`loading footer fills but does not exceed a ${viewport.width}px viewport`, async ({ page }) => {
    await page.setViewportSize(viewport);
    let release;
    const gate = new Promise((resolve) => { release = resolve; });
    await installPublication(page, { beforeResponse: async ({ relative }) => {
      if (relative === 'index.json') await gate;
    } });
    await page.goto('/');
    try {
      await expect(page.getByTestId('dashboard-data-loading')).toBeVisible();
      await page.evaluate(() => document.fonts.ready);
      const geometry = await page.evaluate(() => {
        const footer = document.querySelector('footer').getBoundingClientRect();
        const content = document.querySelector('main > .MuiContainer-root').getBoundingClientRect();
        return { bottom: footer.bottom, top: footer.top, contentBottom: content.bottom, height: document.documentElement.scrollHeight };
      });
      expect(geometry.top).toBeGreaterThanOrEqual(geometry.contentBottom);
      expect(Math.abs(geometry.bottom - viewport.height)).toBeLessThan(1);
      expect(geometry.height).toBe(viewport.height);
    } finally {
      release();
    }
    await ready(page);
    const loaded = await page.evaluate(() => ({
      footerTop: document.querySelector('footer').getBoundingClientRect().top,
      mainBottom: document.querySelector('main').getBoundingClientRect().bottom,
    }));
    expect(loaded.footerTop).toBeGreaterThanOrEqual(loaded.mainBottom);
  });
}

test('footer remains after overflowing content with phone filters expanded', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 600 });
  await openDashboard(page);
  const filters = page.getByRole('button', { name: /^Filters/ });
  await filters.click();
  await expect(filters).toHaveAttribute('aria-expanded', 'true');
  await expect(page.locator('#dashboard-filters .MuiCollapse-root')).toHaveClass(/MuiCollapse-entered/);
  await page.evaluate(() => document.fonts.ready);
  const geometry = await page.evaluate(() => ({
    footerTop: document.querySelector('footer').getBoundingClientRect().top,
    contentBottom: document.querySelector('main > .MuiContainer-root').getBoundingClientRect().bottom,
    height: document.documentElement.scrollHeight,
    width: document.documentElement.scrollWidth,
  }));
  expect(geometry.footerTop).toBeGreaterThanOrEqual(geometry.contentBottom);
  expect(geometry.height).toBeGreaterThan(600);
  expect(geometry.width).toBe(390);
  const footer = page.locator('footer');
  await footer.scrollIntoViewIfNeeded();
  const initial = await measureFooter(page);
  assertVisible(initial);
  await attachGeometry('post-scroll-all-paragraphs', initial);
  await test.info().attach('post-scroll-footer', { body: await page.screenshot(), contentType: 'image/png' });

  // A vertical-only two-pixel clip must fail the bottom guard, not another edge.
  const footerStyle = await footer.getAttribute('style');
  try {
    await footer.evaluate((element) => {
      const bounds = element.getBoundingClientRect();
      Object.assign(element.style, { position: 'fixed', left: `${bounds.left}px`, right: 'auto',
        width: `${bounds.width}px`, height: `${bounds.height}px`, boxSizing: 'border-box',
        top: `${innerHeight - bounds.height + 2}px` });
      window.scrollTo(0, document.scrollingElement.scrollHeight);
    });
    const clipped = await measureFooter(page);
    assertParagraphs(clipped);
    assertFooterFrame(clipped);
    clipped.paragraphs.forEach((p, index) => assertTextInside(clipped, p, index));
    expect(clipped.footer.bottom - clipped.viewport.height).toBeGreaterThanOrEqual(1);
    let rejection;
    try { assertVisible(clipped); } catch (error) { rejection = error.message; }
    await attachGeometry('vertical-clip-negative-proof', { geometry: clipped, rejectedGuard: rejection });
    expect(rejection).toContain('max-scroll bottom residual');
  } finally {
    await restoreStyle(footer, footerStyle);
    expect(await footer.getAttribute('style')).toBe(footerStyle);
  }
  await footer.scrollIntoViewIfNeeded();
  const restoredFooter = await measureFooter(page);
  assertVisible(restoredFooter);
  await attachGeometry('restored-after-vertical-clip', restoredFooter);

  for (const index of [0, 1]) {
    const paragraph = page.locator('footer p').nth(index);
    const original = await paragraph.getAttribute('style');
    try {
      await paragraph.evaluate((element) => {
        const bounds = element.getBoundingClientRect();
        Object.assign(element.style, { position: 'fixed', left: `${bounds.left}px`, right: 'auto',
          width: `${bounds.width}px`, margin: '0', top: `${innerHeight - bounds.height + 2}px` });
        window.scrollTo(0, document.scrollingElement.scrollHeight);
      });
      const clipped = await measureFooter(page);
      assertParagraphs(clipped);
      assertFooterFrame(clipped);
      assertBottom(clipped);
      for (const [other, p] of clipped.paragraphs.entries()) if (other !== index) assertTextInside(clipped, p, other);
      const bounds = clipped.paragraphs[index].bounds;
      expect(bounds.top).toBeGreaterThan(0);
      expect(bounds.top).toBeLessThan(clipped.viewport.height);
      expect(bounds.left).toBeGreaterThan(0);
      expect(bounds.right).toBeLessThan(clipped.viewport.width);
      expect(bounds.bottom - clipped.viewport.height).toBeGreaterThanOrEqual(1);
      let rejection;
      try { assertVisible(clipped); } catch (error) { rejection = error.message; }
      await attachGeometry(`paragraph-${index}-text-negative-proof`, { geometry: clipped, rejectedGuard: rejection });
      expect(rejection).toContain(`paragraph ${index}`);
      expect(rejection).toContain('text bottom');
    } finally {
      await restoreStyle(paragraph, original);
      expect(await paragraph.getAttribute('style')).toBe(original);
    }
    await footer.scrollIntoViewIfNeeded();
    const restored = await measureFooter(page);
    assertVisible(restored);
    await attachGeometry(`restored-after-paragraph-${index}`, restored);
  }
});

for (const width of [320, 390, 768, 1024, 1440, 1920]) {
  test(`footer aligns with content at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page);
    await page.evaluate(() => document.fonts.ready);
    const initial = await measureFooter(page);
    assertAligned(initial);
    await attachGeometry(`alignment-${width}`, initial);
    if (width === 1440) {
      for (const index of [0, 1]) {
        const paragraph = page.locator('footer p').nth(index);
        const original = await paragraph.getAttribute('style');
        try {
          await paragraph.evaluate((element) => { element.style.transform = 'translateX(4px)'; });
          const faulty = await measureFooter(page);
          expect(faulty.scroll.width).toBe(width);
          for (const [other, p] of faulty.paragraphs.entries()) if (other !== index) {
            expect(Math.abs(p.bounds.left - faulty.content.left)).toBeLessThan(1);
            expect(Math.abs(p.bounds.right - faulty.content.right)).toBeLessThan(1);
          }
          expect(Math.abs(faulty.paragraphs[index].bounds.left - faulty.content.left)).toBeGreaterThanOrEqual(1);
          let rejection;
          try { assertAligned(faulty); } catch (error) { rejection = error.message; }
          await attachGeometry(`paragraph-${index}-alignment-negative-proof`, { geometry: faulty, rejectedGuard: rejection });
          expect(rejection).toContain(`paragraph ${index} left alignment`);
        } finally {
          await restoreStyle(paragraph, original);
          expect(await paragraph.getAttribute('style')).toBe(original);
        }
        const restored = await measureFooter(page);
        assertAligned(restored);
        await attachGeometry(`restored-alignment-paragraph-${index}`, restored);
      }
    }
  });
}
