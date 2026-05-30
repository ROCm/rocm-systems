import { test, expect, type Page } from "@playwright/test";

/** Full workflow: create profile → create session → view detail → go back → delete. */
async function createProfile(page: Page, name: string) {
  await page.goto("/profiles");
  await page.getByRole("button", { name: "+ New Profile" }).click();
  await page.locator("#pf-name").fill(name);
  await page.locator("#pf-simulator").selectOption("rocjitsu");
  await page.locator("#pf-gpu").selectOption("MI350X");
  await page.locator("#pf-mode").selectOption("Functional");
  await page.locator("#pf-gpus").fill("8");
  await page.locator("#pf-nodes").fill("4");
  await page.getByRole("button", { name: "Create" }).click();
  await expect(page.locator(".data-table")).toContainText(name);
}

test.describe("End-to-End Workflow", () => {
  test.beforeEach(async ({ request }) => {
    await request.post("http://localhost:50051/api/__reset");
  });

  test("full lifecycle: profile → session → detail → delete", async ({ page }) => {
    // Step 1: Verify overview starts clean
    await page.goto("/");
    await expect(page.locator(".stat-value").nth(1)).toHaveText("0"); // profiles
    await expect(page.locator(".stat-value").nth(2)).toHaveText("0"); // sessions

    // Step 2: Create a profile
    await createProfile(page, "workflow-profile");

    // Step 3: Overview should now show 1 profile
    await page.goto("/");
    await expect(page.locator(".stat-value").nth(1)).toHaveText("1");

    // Step 4: Create a session using that profile
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("workflow-session");
    await page.locator("#sf-profile").selectOption("workflow-profile");
    await page.locator("#sf-image").fill("rocm/pytorch:latest");
    await page.getByRole("button", { name: "Create" }).click();

    // Should navigate to detail
    await expect(page).toHaveURL(/\/sessions\/workflow-session/);
    await expect(page.locator("h2")).toHaveText("workflow-session");
    await expect(page.getByText("rocjitsu", { exact: true })).toBeVisible();
    await expect(page.locator("text=MI350X")).toBeVisible();
    await expect(page.locator("text=8 GPU(s)")).toBeVisible();
    await expect(page.locator("text=4 node(s)")).toBeVisible();

    // Step 5: Overview should show 1 session
    await page.goto("/");
    await expect(page.locator(".stat-value").nth(2)).toHaveText("1");

    // Step 6: Delete the session
    await page.goto("/sessions");
    const sessionRow = page.locator("tr", { hasText: "workflow-session" });
    await sessionRow.getByRole("button", { name: "Delete" }).click();
    await expect(page.locator("text=workflow-session")).not.toBeVisible();

    // Step 7: Overview should show 0 sessions again
    await page.goto("/");
    await expect(page.locator(".stat-value").nth(2)).toHaveText("0");

    // Step 8: Delete the profile
    await page.goto("/profiles");
    const profileRow = page.locator("tr", { hasText: "workflow-profile" });
    await profileRow.getByRole("button", { name: "Delete" }).click();
    await expect(page.locator("text=workflow-profile")).not.toBeVisible();

    // Step 9: Overview back to zeros
    await page.goto("/");
    await expect(page.locator(".stat-value").nth(1)).toHaveText("0");
  });

  test("multiple profiles with different GPUs", async ({ page }) => {
    // Create profile for MI300X
    await page.goto("/profiles");
    await page.getByRole("button", { name: "+ New Profile" }).click();
    await page.locator("#pf-name").fill("mi300x-profile");
    await page.locator("#pf-simulator").selectOption("rocjitsu");
    await page.locator("#pf-gpu").selectOption("MI300X");
    await page.locator("#pf-mode").selectOption("Functional");
    await page.getByRole("button", { name: "Create" }).click();
    await expect(page.locator(".data-table")).toContainText("mi300x-profile");

    // Create profile for MI350X
    await page.getByRole("button", { name: "+ New Profile" }).click();
    await page.locator("#pf-name").fill("mi350x-profile");
    await page.locator("#pf-simulator").selectOption("rocjitsu");
    await page.locator("#pf-gpu").selectOption("MI350X");
    await page.locator("#pf-mode").selectOption("Functional");
    await page.getByRole("button", { name: "Create" }).click();

    // Both should appear
    const table = page.locator(".data-table");
    await expect(table).toContainText("mi300x-profile");
    await expect(table).toContainText("mi350x-profile");
    await expect(table).toContainText("MI300X");
    await expect(table).toContainText("MI350X");
  });

  test("multiple sessions from the same profile", async ({ page }) => {
    await createProfile(page, "multi-sess-profile");

    await page.goto("/sessions");

    // Create first session
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("session-a");
    await page.locator("#sf-profile").selectOption("multi-sess-profile");
    await page.getByRole("button", { name: "Create" }).click();
    await expect(page).toHaveURL(/\/sessions\/session-a/);

    // Create second session
    await page.goto("/sessions");
    await page.getByRole("button", { name: "+ New Session" }).click();
    await page.locator("#sf-name").fill("session-b");
    await page.locator("#sf-profile").selectOption("multi-sess-profile");
    await page.getByRole("button", { name: "Create" }).click();
    await expect(page).toHaveURL(/\/sessions\/session-b/);

    // Both should appear in the list
    await page.goto("/sessions");
    const table = page.locator(".data-table");
    await expect(table).toContainText("session-a");
    await expect(table).toContainText("session-b");

    // Overview should show 2
    await page.goto("/");
    await expect(page.locator(".stat-value").nth(2)).toHaveText("2");
  });
});
