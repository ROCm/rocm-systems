import { expect, test } from '@playwright/test';

for (const [view, label] of [['overview', 'Benchmark set'], ['branch', 'Pull request']]) {
  test(`${label} menu keeps the document scrollbar and layout stable`, async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 500 });
    await page.goto(`/?view=${view}`);
    // MUI's menu hides background controls from the accessibility tree while open.
    const picker = page.getByRole('combobox', { name: label, exact: true, includeHidden: true });
    await expect(picker).toBeVisible();
    await picker.scrollIntoViewIfNeeded();
    const measure = () => page.evaluate(() => ({
      width: document.documentElement.clientWidth,
      overflow: getComputedStyle(document.body).overflow,
      padding: getComputedStyle(document.body).paddingRight,
      scrollY,
      scrollable: document.documentElement.scrollHeight > innerHeight,
      main: document.querySelector('main').getBoundingClientRect().toJSON(),
    }));
    const before = await measure();
    expect(before.scrollable).toBe(true);
    const anchor = await picker.boundingBox();
    await picker.click();
    const listbox = page.getByRole('listbox');
    await expect(listbox).toBeVisible();
    expect(await measure()).toEqual(before);
    const bounds = await picker.boundingBox();
    expect(bounds).toEqual(anchor);
    await page.keyboard.press('Escape');
    await expect(listbox).toBeHidden();
    await expect(picker).toBeFocused();
    expect(await measure()).toEqual(before);
    await picker.click();
    const option = page.getByRole('option').last();
    const selected = await option.innerText();
    await option.click();
    await expect(listbox).toBeHidden();
    await expect(picker).toHaveText(selected);
    await expect(picker).toBeFocused();
  });
}
