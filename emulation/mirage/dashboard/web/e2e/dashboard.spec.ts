import { expect, test, type Page } from "@playwright/test";

const PROFILE = `e2e-profile-${Date.now()}`;

async function dismissAllToasts(page: Page) {
  await page.locator(".toast").evaluateAll((els) => {
    for (const el of els) (el as HTMLElement).click();
  });
}

test.describe.serial("mirage dashboard e2e", () => {
  test("user story 1: overview shows metric cards and detected emulators", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Overview" })).toBeVisible();

    await expect(page.getByTestId("profile-stat")).toBeVisible();
    await expect(page.getByTestId("session-stat")).toBeVisible();
    await expect(page.getByTestId("healthy-stat")).toBeVisible();
    await expect(page.getByTestId("execs-stat")).toBeVisible();

    await expect(page.getByTestId("system-panel")).toBeVisible();
    await expect(page.getByTestId("emulator-row-noop")).toBeVisible();
    await expect(page.getByTestId("emulator-installed-noop")).toContainText(
      /installed/i,
    );

    await expect(page.getByTestId("daemon-version")).toBeVisible();
  });

  test("user story 2: create profile through wizard with dropdown", async ({ page }) => {
    await page.goto("/profiles");

    await expect(page.getByTestId("profiles-empty")).toBeVisible();

    await page.getByTestId("open-profile-wizard").click();
    const wizard = page.getByTestId("profile-wizard");
    await expect(wizard).toBeVisible();

    await page.getByTestId("wizard-name").fill(PROFILE);

    await page.getByTestId("wizard-emulator").click();
    await page.getByTestId("wizard-emulator-option-noop").click();

    await page.getByTestId("wizard-mode-functional").click();
    await page.getByTestId("wizard-nodes").fill("1");
    await page.getByTestId("wizard-gpus").fill("1");
    await page.getByTestId("wizard-description").fill("e2e profile");

    await page.getByTestId("wizard-submit").click();

    await expect(wizard).toHaveCount(0);
    await expect(page.getByTestId(`profile-row-${PROFILE}`)).toBeVisible();

    await expect(page.getByTestId("toast-success")).toBeVisible();
    await dismissAllToasts(page);
  });

  test("user story 3 + 4: start session card, run exec, see streamed output, then stop", async ({ page }) => {
    await page.goto("/sessions");

    await expect(page.getByTestId("no-sessions")).toBeVisible();

    await page.getByTestId("open-start-session").click();
    const startModal = page.getByTestId("start-session-modal");
    await expect(startModal).toBeVisible();

    await page.getByTestId("start-modal-profile").click();
    await page.getByTestId(`start-modal-profile-option-${PROFILE}`).click();
    await page.getByTestId("start-modal-timeout").fill("10");

    await page.getByTestId("start-session-confirm").click();
    await expect(startModal).toHaveCount(0);

    const sessionCard = page.locator('[data-testid^="session-row-"]').first();
    await expect(sessionCard).toBeVisible({ timeout: 30_000 });
    const sessionId = (await sessionCard.getAttribute("data-testid"))!.replace(
      "session-row-",
      "",
    );

    await expect
      .poll(
        async () =>
          (await page
            .getByTestId(`session-healthy-${sessionId}`)
            .textContent()) ?? "",
        { timeout: 30_000 },
      )
      .toContain("healthy");

    await page.getByTestId(`session-open-${sessionId}`).click();
    await expect(page).toHaveURL(new RegExp(`/sessions/${sessionId}$`));

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

    await page.goto("/sessions");
    await page.getByTestId(`stop-session-${sessionId}`).click();
    const confirmStop = page.getByTestId("confirm-stop-session");
    await expect(confirmStop).toBeVisible();
    await page.getByTestId("confirm-stop-session-confirm").click();
    await expect(page.getByTestId(`session-row-${sessionId}`)).toHaveCount(0, {
      timeout: 15_000,
    });
    await dismissAllToasts(page);
  });

  test("user story 5: delete the profile via confirmation modal", async ({ page }) => {
    await page.goto("/profiles");
    await page.getByTestId(`delete-profile-${PROFILE}`).click();
    const confirm = page.getByTestId("confirm-delete-profile");
    await expect(confirm).toBeVisible();
    await page.getByTestId("confirm-delete-profile-confirm").click();
    await expect(page.getByTestId(`profile-row-${PROFILE}`)).toHaveCount(0);
  });
});
