import { test, expect, type Page } from "@playwright/test";

/** Helper: create a profile and session for terminal tests. */
async function setupSession(page: Page) {
  // Create profile
  await page.goto("/profiles");
  await page.getByRole("button", { name: "+ New Profile" }).click();
  await page.locator("#pf-name").fill("run-profile");
  await page.locator("#pf-simulator").selectOption("rocjitsu");
  await page.locator("#pf-gpu").selectOption("MI300X");
  await page.locator("#pf-mode").selectOption("Functional");
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.locator(".data-table")).toContainText("run-profile");

  // Create session
  await page.goto("/sessions");
  await page.getByRole("button", { name: "+ New Session" }).click();
  await page.locator("#sf-name").fill("run-session");
  await page.locator("#sf-profile").selectOption("run-profile");
  await page.locator("#sf-image").fill("ubuntu:22.04");
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page).toHaveURL(/\/sessions\/run-session/);
}

test.describe("Runs Page", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("http://localhost:50051/api/__reset");
  });
  test("initially shows no terminals", async ({ page }) => {
    await page.goto("/runs");
    await expect(page.locator("h2")).toHaveText("Runs");
    await expect(page.locator(".empty")).toContainText("No terminals");
  });

  test("new terminal button opens form", async ({ page }) => {
    await page.goto("/runs");
    await page.getByRole("button", { name: "+ New Terminal" }).click();
    await expect(page.locator(".create-form")).toBeVisible();
    await expect(page.locator("#tf-session")).toBeVisible();
  });

  test("cancel hides the terminal form", async ({ page }) => {
    await page.goto("/runs");
    await page.getByRole("button", { name: "+ New Terminal" }).click();
    await expect(page.locator(".create-form")).toBeVisible();
    await page.getByRole("button", { name: "Cancel" }).click();
    await expect(page.locator(".create-form")).not.toBeVisible();
  });

  test("create terminal requires a running session", async ({ page }) => {
    await setupSession(page);
    await page.goto("/runs");
    await page.getByRole("button", { name: "+ New Terminal" }).click();

    // Should show the running session in dropdown
    const select = page.locator("#tf-session");
    const options = select.locator("option");
    // At least the placeholder + 1 session
    await expect(options).toHaveCount(2);
    await expect(select).toContainText("run-session");
  });
});

test.describe("Navigation", () => {
  test("sidebar navigation links work", async ({ page }) => {
    await page.goto("/");

    // Overview
    await expect(page.locator("h2")).toHaveText("Overview");

    // Simulators
    await page.locator("nav").getByText("Simulators").click();
    await expect(page).toHaveURL(/\/simulators/);
    await expect(page.locator("h2")).toHaveText("Simulators");

    // Profiles
    await page.locator("nav").getByText("Profiles").click();
    await expect(page).toHaveURL(/\/profiles/);
    await expect(page.locator("h2")).toHaveText("Profiles");

    // Sessions
    await page.locator("nav").getByText("Sessions").click();
    await expect(page).toHaveURL(/\/sessions/);
    await expect(page.locator("h2")).toHaveText("Sessions");

    // Runs
    await page.locator("nav").getByText("Runs").click();
    await expect(page).toHaveURL(/\/runs/);
    await expect(page.locator("h2")).toHaveText("Runs");

    // Topology Editor
    await page.locator("nav").getByText("Topology Editor").click();
    await expect(page).toHaveURL(/\/topology/);
  });

  test("sidebar shows Mirage branding", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".sidebar-header h1")).toHaveText("Mirage");
    await expect(page.locator(".subtitle")).toContainText("AMD GPU Simulator");
  });

  test("active nav link is highlighted", async ({ page }) => {
    await page.goto("/simulators");
    const activeLink = page.locator("nav a.active");
    await expect(activeLink).toHaveText("Simulators");
  });
});
