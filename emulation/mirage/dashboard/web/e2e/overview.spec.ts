import { test, expect } from "@playwright/test";

test.describe("Overview Page", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("http://localhost:50051/api/__reset");
  });

  test("shows overview stats cards", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("h2")).toHaveText("Overview");
    // Should show 3 stat cards
    const cards = page.locator(".stat-card");
    await expect(cards).toHaveCount(3);

    // Initially: 1 simulator, 0 profiles, 0 sessions
    const values = page.locator(".stat-value");
    await expect(values.nth(0)).toHaveText("1"); // simulators
    await expect(values.nth(1)).toHaveText("0"); // profiles
    await expect(values.nth(2)).toHaveText("0"); // sessions
  });

  test("stat labels are correct", async ({ page }) => {
    await page.goto("/");
    const labels = page.locator(".stat-label");
    await expect(labels.nth(0)).toHaveText("Simulators");
    await expect(labels.nth(1)).toHaveText("Profiles");
    await expect(labels.nth(2)).toHaveText("Sessions");
  });
});
