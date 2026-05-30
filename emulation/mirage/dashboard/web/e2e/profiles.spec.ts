import { test, expect } from "@playwright/test";

test.describe("Profile CRUD", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("http://localhost:50051/api/__reset");
  });

  test("initially shows no profiles", async ({ page }) => {
    await page.goto("/profiles");
    await expect(page.locator("h2")).toHaveText("Profiles");
    await expect(page.locator(".empty")).toHaveText("No profiles.");
  });

  test("create a profile via the form", async ({ page }) => {
    await page.goto("/profiles");

    // Open form
    await page.getByRole("button", { name: "+ New Profile" }).click();

    // Fill in name
    await page.locator("#pf-name").fill("test-profile");

    // Select simulator
    await page.locator("#pf-simulator").selectOption("rocjitsu");

    // Select GPU (enabled after simulator is selected)
    await page.locator("#pf-gpu").selectOption("MI300X");

    // Select mode
    await page.locator("#pf-mode").selectOption("Functional");

    // Set GPU count
    await page.locator("#pf-gpus").fill("4");

    // Set node count
    await page.locator("#pf-nodes").fill("2");

    // Submit
    await page.getByRole("button", { name: "Create" }).click();

    // Form should close, profile should appear in table
    const table = page.locator(".data-table");
    await expect(table).toBeVisible();
    await expect(table).toContainText("test-profile");
    await expect(table).toContainText("rocjitsu");
    await expect(table).toContainText("MI300X");
    await expect(table).toContainText("Functional");
    await expect(table).toContainText("4");
    await expect(table).toContainText("2");
  });

  test("delete a profile", async ({ page }) => {
    // First create a profile
    await page.goto("/profiles");
    await page.getByRole("button", { name: "+ New Profile" }).click();
    await page.locator("#pf-name").fill("to-delete");
    await page.locator("#pf-simulator").selectOption("rocjitsu");
    await page.locator("#pf-gpu").selectOption("MI325X");
    await page.locator("#pf-mode").selectOption("Functional");
    await page.getByRole("button", { name: "Create" }).click();
    await expect(page.locator(".data-table")).toContainText("to-delete");

    // Delete it
    const row = page.locator("tr", { hasText: "to-delete" });
    await row.getByRole("button", { name: "Delete" }).click();

    // Should be gone
    await expect(page.locator("text=to-delete")).not.toBeVisible();
  });

  test("shows validation error for duplicate name", async ({ page }) => {
    await page.goto("/profiles");

    // Create first profile
    await page.getByRole("button", { name: "+ New Profile" }).click();
    await page.locator("#pf-name").fill("dup-profile");
    await page.locator("#pf-simulator").selectOption("rocjitsu");
    await page.locator("#pf-gpu").selectOption("MI300X");
    await page.locator("#pf-mode").selectOption("Functional");
    await page.getByRole("button", { name: "Create" }).click();
    await expect(page.locator(".data-table")).toContainText("dup-profile");

    // Try to create duplicate
    await page.getByRole("button", { name: "+ New Profile" }).click();
    await page.locator("#pf-name").fill("dup-profile");
    await page.locator("#pf-simulator").selectOption("rocjitsu");
    await page.locator("#pf-gpu").selectOption("MI300X");
    await page.locator("#pf-mode").selectOption("Functional");
    await page.getByRole("button", { name: "Create" }).click();

    // Should show an error
    await expect(page.locator(".error")).toContainText("already exists");
  });

  test("cancel hides the form", async ({ page }) => {
    await page.goto("/profiles");
    await page.getByRole("button", { name: "+ New Profile" }).click();
    await expect(page.locator(".create-form")).toBeVisible();

    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.locator(".create-form")).not.toBeVisible();
  });
});
