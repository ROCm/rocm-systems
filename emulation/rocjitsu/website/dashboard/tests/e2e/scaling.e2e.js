import { expect, test } from '@playwright/test';

// Screen and viewport are CSS pixels at 100% browser zoom; model OS scaling
// explicitly rather than treating the monitor's resolution as innerWidth.
const displays = [
  { name: '1080p', screen: { width: 1280, height: 720 }, viewport: { width: 1280, height: 640 }, pixelRatio: 1.5 },
  { name: '1440p', screen: { width: 1707, height: 960 }, viewport: { width: 1707, height: 880 }, pixelRatio: 1.5 },
  { name: '4K', screen: { width: 2560, height: 1440 }, viewport: { width: 2560, height: 1360 }, pixelRatio: 1.5 },
  { name: 'ultrawide 1440p', screen: { width: 2293, height: 960 }, viewport: { width: 2293, height: 880 }, pixelRatio: 1.5 },
  { name: 'reported ultrawide viewport', screen: { width: 1835, height: 768 }, viewport: { width: 1439, height: 643 }, pixelRatio: 1.875 },
];

for (const display of displays) {
  test(`${display.name} uses CSS viewport width and keeps cards beside the sidebar`, async ({ browser }) => {
    const context = await browser.newContext({
      screen: display.screen, viewport: display.viewport, deviceScaleFactor: display.pixelRatio,
    });
    try {
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.goto('/');
      const expectedScale = 0.7392;
      const zoom = () => page.locator('body').evaluate((body) => Number(getComputedStyle(body).zoom));
      await expect.poll(zoom).toBeCloseTo(expectedScale, 5);
      await expect(page.getByTestId('performance-trend')).toBeVisible();
      const geometry = await page.evaluate(() => {
        const container = document.querySelector('main > .MuiContainer-root');
        const bounds = container.getBoundingClientRect();
        return {
          gutter: bounds.left - document.querySelector('aside').getBoundingClientRect().right,
          unusedWidth: document.querySelector('main').getBoundingClientRect().width - bounds.width,
          overflow: document.documentElement.scrollWidth > innerWidth,
        };
      });
      expect(Math.abs(geometry.gutter)).toBeLessThan(1);
      expect(Math.abs(geometry.unusedWidth)).toBeLessThan(1);
      expect(geometry.overflow).toBe(false);

      const picker = page.getByRole('combobox').first();
      const anchor = await picker.boundingBox();
      await picker.click();
      const listbox = page.getByRole('listbox');
      await expect(listbox).toBeVisible();
      const menu = await listbox.evaluate((element) => ({
        zoom: Number(getComputedStyle(element.closest('.MuiPopover-root')).zoom),
        fontSize: parseFloat(getComputedStyle(element.querySelector('[role="option"]')).fontSize),
      }));
      expect(menu.zoom * expectedScale).toBeCloseTo(1, 4);
      expect(menu.fontSize).toBeCloseTo(14 * expectedScale, 2);
      await expect.poll(async () => Math.abs((await listbox.boundingBox()).x - anchor.x)).toBeLessThan(4);
      await page.keyboard.press('Escape');
      await expect(listbox).toBeHidden();

      // Playwright's setViewportSize also changes screen dimensions. Keep the
      // monitor fixed while modeling a real browser-window height change.
      const cdp = await context.newCDPSession(page);
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width: display.viewport.width, height: 600,
        screenWidth: display.screen.width, screenHeight: display.screen.height,
        deviceScaleFactor: display.pixelRatio, mobile: false,
      });
      await expect.poll(zoom).toBeCloseTo(expectedScale, 5);
      expect(errors).toEqual([]);
    } finally {
      await context.close();
    }
  });
}

for (const screen of [{ width: 1920, height: 1080 }, { width: 3840, height: 2160 }]) {
  test(`${screen.width}px display follows CSS width through modeled browser zoom and reload`, async ({ browser }) => {
    const context = await browser.newContext({ screen, viewport: screen, deviceScaleFactor: 1 });
    try {
      const page = await context.newPage();
      await page.goto('/');
      const zoom = () => page.locator('body').evaluate((body) => Number(getComputedStyle(body).zoom));
      const expectedScale = screen.width === 1920 ? 0.7392 : 1.064448;
      await expect.poll(zoom).toBeCloseTo(expectedScale, 5);
      const cdp = await context.newCDPSession(page);
      // Keep physical viewport width fixed while CSS width shrinks and DPR rises.
      // This is metric emulation, not a native toolbar-zoom test.
      for (const [factor, zoomedScale] of screen.width === 1920
        ? [[1.25, 0.7392], [1.5, 0.7392], [2, 0.7392], [1, 0.7392]]
        : [[1.25, 0.8515584], [1.5, 0.7392], [2, 0.7392], [1, 1.064448]]) {
        await cdp.send('Emulation.setDeviceMetricsOverride', {
          width: screen.width / factor, height: screen.height / factor,
          screenWidth: screen.width, screenHeight: screen.height,
          deviceScaleFactor: factor, mobile: false,
        });
        await expect.poll(zoom).toBeCloseTo(zoomedScale, 5);
        await page.reload();
        await expect.poll(zoom).toBeCloseTo(zoomedScale, 5);
      }
    } finally {
      await context.close();
    }
  });
}

test('desktop resizing follows one continuous width formula', async ({ browser }) => {
  const context = await browser.newContext({
    screen: { width: 3440, height: 1440 }, viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1.5,
  });
  try {
    const page = await context.newPage();
    await page.goto('/');
    const cdp = await context.newCDPSession(page);
    for (const [width, expectedScale] of [[1280, 0.7392], [1920, 0.7392], [2560, 0.7392], [3000, 0.8316], [3440, 0.953568], [1920, 0.7392]]) {
      await cdp.send('Emulation.setDeviceMetricsOverride', {
        width, height: 900, screenWidth: 3440, screenHeight: 1440,
        deviceScaleFactor: 1.5, mobile: false,
      });
      await expect.poll(() => page.locator('body').evaluate((body) => Number(getComputedStyle(body).zoom)))
        .toBeCloseTo(expectedScale, 5);
    }
  } finally {
    await context.close();
  }
});

for (const pixelRatio of [1, 2, 3]) {
  test(`1920 CSS pixels keeps the same visible text and layout at DPR ${pixelRatio}`, async ({ browser }) => {
    const context = await browser.newContext({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: pixelRatio });
    try {
      const page = await context.newPage();
      await page.goto('/');
      await expect(page.getByTestId('performance-trend')).toBeVisible();
      const geometry = await page.evaluate(async () => {
        await document.fonts.ready;
        const probe = document.createElement('span');
        probe.textContent = 'Dashboard text';
        probe.style.cssText = 'display:inline-block;font-size:14px;line-height:14px';
        document.body.append(probe);
        const textHeight = probe.getBoundingClientRect().height;
        probe.remove();
        return { textHeight, layoutWidth: document.body.clientWidth };
      });
      // Visible CSS geometry is the acceptance target, not a copy of the formula.
      expect(geometry.textHeight).toBeGreaterThan(10.3);
      expect(geometry.textHeight).toBeLessThan(10.4);
      expect(geometry.layoutWidth).toBeGreaterThan(2596);
      expect(geometry.layoutWidth).toBeLessThan(2599);
    } finally {
      await context.close();
    }
  });
}

test('popup geometry and text follow applied density, including CSS fallback', async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto('/');
  await expect(page.getByTestId('performance-trend')).toBeVisible();
  await page.evaluate(() => document.documentElement.style.removeProperty('--dashboard-scale'));
  await expect(page.locator('body')).toHaveCSS('zoom', '0.7392');
  await page.evaluate(() => document.body.style.setProperty('--dashboard-density', '0.9'));
  await expect(page.locator('body')).toHaveCSS('zoom', '0.9');
  const picker = page.getByRole('combobox').first();
  const anchor = await picker.boundingBox();
  await picker.click();
  const listbox = page.getByRole('listbox');
  await expect(listbox).toBeVisible();
  const popup = await listbox.evaluate((element) => ({
    zoom: Number(getComputedStyle(element.closest('.MuiPopover-root')).zoom),
    fontSize: parseFloat(getComputedStyle(element.querySelector('[role="option"]')).fontSize),
  }));
  expect(popup.zoom).toBeCloseTo(1.111111, 4);
  expect(popup.fontSize).toBeCloseTo(12.6, 2);
  await expect.poll(async () => Math.abs((await listbox.boundingBox()).x - anchor.x)).toBeLessThan(4);
});

test('phones remain native-sized rather than receiving desktop scaling', async ({ browser }) => {
  const context = await browser.newContext({
    viewport: { width: 390, height: 844 }, deviceScaleFactor: 3,
  });
  try {
    const page = await context.newPage();
    await page.goto('/');
    await expect(page.locator('body')).toHaveCSS('zoom', '1');
    await expect(page.getByTestId('dashboard-wordmark')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  } finally {
    await context.close();
  }
});
