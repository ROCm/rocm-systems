import { test, expect, type Page } from "@playwright/test";

/** Helper: create a profile so sessions can reference it. */
async function createProfile(page: Page, name = "e2e-profile") {
  await page.goto("/profiles");
  await page.getByRole("button", { name: "+ New Profile" }).click();
  await page.locator("#pf-name").fill(name);
  await page.locator("#pf-simulator").selectOption("rocjitsu");
  await page.locator("#pf-gpu").selectOption("MI300X");
  await page.locator("#pf-mode").selectOption("Functional");
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.locator(".data-table")).toContainText(name);
}

test.describe("Session CRUD", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("http://localhost:50051/api/__reset");
  });

  test.beforeEach(async ({ page }) => {
    await createProfile(page, "sess-profile");
  });

  test("initially shows no sessions (before profile created)", async ({ page }) => {
    // Reset again to clear the profile we just made
    await page.request.post("http://localhost:50051/api/__reset");
    await page.goto("/sessions");
    await expect(page.locator("h2")).toHaveText("Sessions");
    await expect(page.locator(".empty")).toHaveText("No sessions.");
  });

  test("create a session", async ({ page }) => {
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();

    await page.locator("#sf-name").fill("test-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.locator("#sf-image").fill("pytorch:latest");
    await page.getByRole("button", { name: "Create" }).click();

    // Should navigate to session detail
    await expect(page).toHaveURL(/\/sessions\/test-session/);
    await expect(page.locator("h2")).toHaveText("test-session");
  });

  test("session detail shows profile and simulator info", async ({ page }) => {
    // Create session
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("detail-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.locator("#sf-image").fill("rocm:6.0");
    await page.getByRole("button", { name: "Create" }).click();

    await expect(page).toHaveURL(/\/sessions\/detail-session/);

    // Check detail fields
    await expect(page.getByText("rocjitsu", { exact: true })).toBeVisible();
    await expect(page.getByText("sess-profile", { exact: true })).toBeVisible();
    await expect(page.locator("text=rocm:6.0")).toBeVisible();
  });

  test("session detail shows performance stats", async ({ page }) => {
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("perf-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.getByRole("button", { name: "Create" }).click();

    await expect(page).toHaveURL(/\/sessions\/perf-session/);

    // Performance section
    await expect(page.locator("h3").filter({ hasText: "Performance" })).toBeVisible();

    const statLabels = page.locator(".stat-label");
    await expect(statLabels.filter({ hasText: "Uptime" })).toBeVisible();
    await expect(statLabels.filter({ hasText: "Ticks" })).toBeVisible();
    await expect(statLabels.filter({ hasText: "IPC" })).toBeVisible();
    await expect(statLabels.filter({ hasText: "Speed" })).toBeVisible();
    await expect(statLabels.filter({ hasText: "Active Contexts" })).toBeVisible();
  });

  test("session detail shows docker output log", async ({ page }) => {
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("log-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.getByRole("button", { name: "Create" }).click();

    await expect(page).toHaveURL(/\/sessions\/log-session/);

    // Should have docker output section
    await expect(page.locator("h3").filter({ hasText: "Docker Output" })).toBeVisible();
    await expect(page.locator(".session-log")).toBeVisible();
    await expect(page.locator(".session-log")).toContainText("Session 'log-session'");
    await expect(page.locator(".session-log")).toContainText("Simulator: rocjitsu");
  });

  test("delete a session from the list", async ({ page }) => {
    // Create a session
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("del-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.getByRole("button", { name: "Create" }).click();

    // Go back to sessions list
    await page.goto("/sessions");
    await expect(page.locator(".data-table")).toContainText("del-session");

    // Delete
    const row = page.locator("tr", { hasText: "del-session" });
    await row.getByRole("button", { name: "Delete" }).click();

    await expect(page.locator("text=del-session")).not.toBeVisible();
  });

  test("session list shows health badges", async ({ page }) => {
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("health-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.getByRole("button", { name: "Create" }).click();

    await page.goto("/sessions");
    await expect(page.locator(".status-badge")).toBeVisible();
    await expect(page.locator(".status-badge")).toHaveText("Healthy");
  });

  test("session detail back link works", async ({ page }) => {
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("back-session");
    await page.locator("#sf-profile").selectOption("sess-profile");
    await page.getByRole("button", { name: "Create" }).click();

    await expect(page).toHaveURL(/\/sessions\/back-session/);
    await page.locator(".back-link").click();
    await expect(page).toHaveURL(/\/sessions$/);
  });

  test("reject session creation without a profile", async ({ page }) => {
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("no-profile-session");
    // Form validation prevents submission (required select)
    const profileSelect = page.locator("#sf-profile");
    await expect(profileSelect).toHaveAttribute("required", "");
  });
});
