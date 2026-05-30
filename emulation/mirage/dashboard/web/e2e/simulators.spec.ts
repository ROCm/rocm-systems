import { test, expect } from "@playwright/test";

test.describe("Simulator List Page", () => {
  test("lists the builtin rocjitsu simulator", async ({ page }) => {
    await page.goto("/simulators");
    await expect(page.locator("h2")).toHaveText("Simulators");

    const card = page.locator(".card").first();
    await expect(card).toBeVisible();
    await expect(card.locator("h3")).toHaveText("rocjitsu");
    await expect(card.locator(".version")).toHaveText("v0.5.0");
    await expect(card).toContainText("AMD CDNA functional simulator");
    await expect(card).toContainText("3 GPUs");
    await expect(card).toContainText("Functional");
  });

  test("clicking a simulator navigates to detail page", async ({ page }) => {
    await page.goto("/simulators");
    await page.locator(".card").first().click();
    await expect(page).toHaveURL(/\/simulators\/rocjitsu/);
    await expect(page.locator("h2")).toHaveText("rocjitsu");
  });
});

test.describe("Simulator Detail Page", () => {
  test("shows simulator details", async ({ page }) => {
    await page.goto("/simulators/rocjitsu");
    await expect(page.locator("h2")).toHaveText("rocjitsu");
    await expect(page.locator(".version")).toHaveText("v0.5.0");
    await expect(page.locator("text=AMD CDNA functional simulator")).toBeVisible();
  });

  test("shows supported GPUs table", async ({ page }) => {
    await page.goto("/simulators/rocjitsu");
    await expect(page.locator("h3").filter({ hasText: "Supported GPUs" })).toBeVisible();

    const table = page.locator(".data-table");
    await expect(table).toBeVisible();

    // Should show MI300X, MI325X, MI350X
    await expect(table).toContainText("MI300X");
    await expect(table).toContainText("MI325X");
    await expect(table).toContainText("MI350X");
    await expect(table).toContainText("gfx942");
    await expect(table).toContainText("gfx950");
  });

  test("shows simulator mode and custom GPU support", async ({ page }) => {
    await page.goto("/simulators/rocjitsu");
    await expect(page.getByText("Functional", { exact: true })).toBeVisible();
    await expect(page.locator("text=Not supported")).toBeVisible(); // custom GPUs
  });

  test("back link navigates to simulator list", async ({ page }) => {
    await page.goto("/simulators/rocjitsu");
    await page.locator(".back-link").click();
    await expect(page).toHaveURL(/\/simulators$/);
  });
});
