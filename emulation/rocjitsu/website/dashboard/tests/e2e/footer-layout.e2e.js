import { expect, test } from '@playwright/test';
import { installPublication, openDashboard, ready } from './helpers/dashboard.js';

for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 1800 }]) {
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
  await page.locator('footer').scrollIntoViewIfNeeded();
  await expect(page.locator('footer')).toBeInViewport({ ratio: 1 });
});

for (const width of [320, 390, 768, 1024, 1440, 1920]) {
  test(`footer aligns with content at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await openDashboard(page);
    await page.evaluate(() => document.fonts.ready);
    const geometry = await page.evaluate(() => {
      const container = document.querySelector('main > .MuiContainer-root');
      const bounds = container.getBoundingClientRect();
      const style = getComputedStyle(container);
      const text = document.querySelector('footer p').getBoundingClientRect();
      return {
        left: bounds.left + parseFloat(style.paddingLeft),
        right: bounds.right - parseFloat(style.paddingRight),
        textLeft: text.left,
        textRight: text.right,
        scrollWidth: document.documentElement.scrollWidth,
      };
    });
    expect(Math.abs(geometry.textLeft - geometry.left)).toBeLessThan(1);
    expect(Math.abs(geometry.textRight - geometry.right)).toBeLessThan(1);
    expect(geometry.scrollWidth).toBe(width);
  });
}
