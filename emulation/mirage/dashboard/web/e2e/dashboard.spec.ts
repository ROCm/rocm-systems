import { expect, test } from "@playwright/test";

const PROFILE = `e2e-profile-${Date.now()}`;

test.describe.serial("mirage dashboard e2e", () => {
  test("overview page loads", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();
    await expect(page.getByTestId("profile-stat")).toBeVisible();
    await expect(page.getByTestId("session-stat")).toBeVisible();
  });

  test("create a profile via the UI", async ({ page }) => {
    await page.goto("/profiles");
    await page.getByTestId("new-profile-name").fill(PROFILE);
    await page.getByTestId("submit-profile").click();
    await expect(page.getByTestId(`profile-row-${PROFILE}`)).toBeVisible();
  });

  test("start a session, run an exec, see streamed output", async ({
    page,
  }) => {
    await page.goto("/sessions");
    await page.getByTestId("new-session-profile").selectOption(PROFILE);
    await page.getByTestId("submit-session").click();

    const sessionRow = page
      .locator('[data-testid^="session-row-"]')
      .first();
    await expect(sessionRow).toBeVisible({ timeout: 30_000 });
    const sessionId = (await sessionRow.getAttribute("data-testid"))!.replace(
      "session-row-",
      "",
    );

    // Wait for health to flip to healthy before exec
    await expect
      .poll(
        async () => {
          const text = await sessionRow.textContent();
          return text?.includes("healthy") ? "healthy" : "starting";
        },
        { timeout: 20_000 },
      )
      .toBe("healthy");

    await page.goto(`/sessions/${sessionId}`);
    await page
      .getByTestId("exec-command")
      .fill("/bin/sh -c 'echo hello-mirage'");
    await page.getByTestId("submit-exec").click();

    await expect(page.getByTestId("attach-output")).toContainText(
      "hello-mirage",
      { timeout: 30_000 },
    );
    await expect(page.getByTestId("attach-exit")).toBeVisible({
      timeout: 15_000,
    });

    // Tear down session via sessions list
    await page.goto("/sessions");
    page.on("dialog", (d) => d.accept());
    await page.getByTestId(`stop-session-${sessionId}`).click();
    await expect(page.getByTestId(`session-row-${sessionId}`)).toHaveCount(0, {
      timeout: 15_000,
    });
  });

  test("delete the profile", async ({ page }) => {
    await page.goto("/profiles");
    page.on("dialog", (d) => d.accept());
    await page.getByTestId(`delete-profile-${PROFILE}`).click();
    await expect(page.getByTestId(`profile-row-${PROFILE}`)).toHaveCount(0);
  });
});
