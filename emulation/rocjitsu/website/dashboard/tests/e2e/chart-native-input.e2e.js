import { expect, test } from '@playwright/test';
import { readChart } from './helpers/chart.js';
import { ensureGridBenchmark, openDashboard } from './helpers/dashboard.js';

test('chart native wheel/buttons, global release and owned listener teardown stay coherent in both themes', async ({ page }) => {
  await openDashboard(page, '/?view=benchmarks');
  const cdp = await page.context().newCDPSession(page);
  try {
    for (const theme of ['light', 'dark']) {
      if (theme === 'dark') await page.getByRole('button', { name: 'Use dark theme', exact: true }).click();
      await page.getByRole('tab', { name: 'Benchmarks', exact: true }).click();
      await ensureGridBenchmark(page, 'Fictional GEMM');
      const chart = page.getByRole('img', { name: 'Fictional GEMM duration history', exact: true });
      await chart.scrollIntoViewIfNeeded();
      await readChart(chart, (instance) => {
        const zr = instance.getZr(); const viewport = zr.painter.getViewportRoot(); const events = [];
        const handlers = ['mousedown', 'mouseup', 'mousewheel', 'globalout'].map((type) => {
          const listener = (event) => events.push({ type, offsetX: event?.offsetX, offsetY: event?.offsetY,
            which: event?.which, zrDelta: event?.wheelDelta, capturing: zr.handler.proxy.__pointerCapturing,
            raw: event?.event && { which: event.event.which, button: event.event.button, zrDelta: event.event.zrDelta,
              clientX: event.event.clientX, clientY: event.event.clientY, targetIsViewport: event.event.target === viewport },
            bounds: viewport.getBoundingClientRect().toJSON(), width: instance.getWidth(), height: instance.getHeight() });
          zr.on(type, listener); return [type, listener];
        });
        window.__chartInputProbe = { viewport, events, proxy: zr.handler.proxy,
          cleanup: () => handlers.forEach(([type, listener]) => zr.off(type, listener)) };
      });
      const object = await cdp.send('Runtime.evaluate', { expression: 'window.__chartInputProbe.viewport' });
      const listeners = await cdp.send('DOMDebugger.getEventListeners', { objectId: object.result.objectId });
      const captures = listeners.listeners.filter((listener) => listener.useCapture && listener.passive).map((listener) => listener.type);
      expect(captures.length).toBeGreaterThan(0);
      expect(new Set(captures).size).toBe(captures.length);

      await chart.hover();
      await page.mouse.wheel(0, 120);
      await expect.poll(() => page.evaluate(() => window.__chartInputProbe.events.some((e) => e.type === 'mousewheel'))).toBe(true);
      const wheel = await page.evaluate(() => window.__chartInputProbe.events.find((e) => e.type === 'mousewheel'));
      expect(Number.isFinite(wheel.raw.zrDelta)).toBe(true);
      expect(wheel.raw.zrDelta).toBeLessThan(0);

      await chart.scrollIntoViewIfNeeded();
      const box = await chart.boundingBox();
      const x = box.x + box.width / 2; const y = box.y + box.height / 2;
      await page.mouse.move(x, y);
      await page.mouse.down();
      await page.mouse.move(x + 10, y + 10, { steps: 2 });
      await expect.poll(() => page.evaluate(() => window.__chartInputProbe.proxy.__pointerCapturing)).toBe(true);
      await page.mouse.move(0, 0, { steps: 2 });
      await page.mouse.up();
      await expect.poll(() => page.evaluate(() => window.__chartInputProbe.proxy.__pointerCapturing)).toBe(false);
      const events = await page.evaluate(() => window.__chartInputProbe.events);
      const down = events.find((e) => e.type === 'mousedown');
      expect(down.raw.button).toBe(0);
      expect(down.raw.which).toBe(1);
      const up = events.findLast((e) => e.type === 'mouseup');
      expect(up.raw.targetIsViewport).toBe(true); // Installed global FakeGlobalEvent path.
      expect(up.offsetX).toBeLessThan(0);
      expect(up.offsetY).toBeLessThan(0);
      expect(up.capturing).toBe(false);
      expect(Math.abs(up.offsetX - (up.raw.clientX - up.bounds.left) * up.width / up.bounds.width)).toBeLessThan(1);
      expect(Math.abs(up.offsetY - (up.raw.clientY - up.bounds.top) * up.height / up.bounds.height)).toBeLessThan(1);
      expect(events.some((e) => e.type === 'globalout')).toBe(true);
      await test.info().attach(`native-input-${theme}`, { body: JSON.stringify({ events, captures }, null, 2), contentType: 'application/json' });

      await page.evaluate(() => window.__chartInputProbe.cleanup());
      await page.getByRole('tab', { name: 'Overview', exact: true }).click();
      const removed = await cdp.send('DOMDebugger.getEventListeners', { objectId: object.result.objectId });
      expect(removed.listeners.filter((listener) => listener.useCapture)).toHaveLength(0);
      await cdp.send('Runtime.releaseObject', { objectId: object.result.objectId });
      await page.evaluate(() => { delete window.__chartInputProbe; });

      const trend = page.getByTestId('performance-trend');
      const canvas = trend.getByRole('img');
      await canvas.scrollIntoViewIfNeeded();
      const point = await readChart(canvas, (instance) => {
        const last = instance.getOption().series[0].data.findLast((p) => Number.isFinite(p[1]));
        return { pixel: instance.convertToPixel({ seriesIndex: 0 }, last), width: instance.getWidth(), height: instance.getHeight() };
      });
      const bounds = await canvas.boundingBox();
      // ALL ends exactly on the plot border. Aim half a logical pixel inside
      // that slot so CSS-scaled coordinate rounding cannot put it outside grid.
      await page.mouse.move(bounds.x + (point.pixel[0] - 0.5) * bounds.width / point.width, bounds.y + point.pixel[1] * bounds.height / point.height);
      await expect(trend.getByTestId('point-hover-guide')).toBeVisible();
      await expect(canvas.getByText(/^Commit [a-f0-9]{8}$/)).toBeVisible();
      await test.info().attach(`hover-${theme}`, { body: await page.screenshot(), contentType: 'image/png' });
      await page.mouse.move(0, 0);
      await expect(trend.getByTestId('point-hover-guide')).not.toBeVisible();
      const keyboard = trend.getByRole('group', { name: 'Inspect performance trend', exact: true });
      await keyboard.press('End');
      await expect(trend.getByRole('status')).toContainText('Attempt fictional-develop-23');
      await keyboard.press('Escape');
      await expect(trend.getByRole('status')).toHaveCount(0);
    }
  } finally {
    await page.evaluate(() => { window.__chartInputProbe?.cleanup(); delete window.__chartInputProbe; });
    await cdp.detach();
  }
});
