import { expect } from '@playwright/test';

export async function expectDialogTypographyContained(dialog) {
  const overflowingText = await dialog.evaluate((root) => [...root.querySelectorAll('.MuiTypography-root')]
    .filter((element) => element.textContent.trim())
    .flatMap((element) => {
      const elementBounds = element.getBoundingClientRect();
      const range = document.createRange();
      range.selectNodeContents(element);
      const textBounds = range.getBoundingClientRect();
      return textBounds.top < elementBounds.top - 0.1 || textBounds.bottom > elementBounds.bottom + 0.1
        ? [element.textContent.trim().replace(/\s+/g, ' ')]
        : [];
    }));
  expect(overflowingText).toEqual([]);
}
