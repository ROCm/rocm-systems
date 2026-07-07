import { expect, test, type Page } from "@playwright/test";

const PROFILE = `e2e-profile-${Date.now()}`;
const TOPOLOGY = `e2e-topo-${Date.now()}`;
const PROFILE_WITH_TOPO = `e2e-profile-topo-${Date.now()}`;

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

  test("user story 4b: terminal stdin is forwarded and a running exec can be killed", async ({ page }) => {
    await page.goto("/sessions");

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

    // `cat` echoes whatever it reads on stdin back to stdout, so it is a
    // perfect probe for the "is terminal input forwarded?" question and it
    // stays running until killed.
    await page.getByTestId("exec-command").fill("/bin/cat");
    await page.getByTestId("submit-exec").click();

    // The new exec row should appear with a Kill button (it is running).
    const killButton = page.locator('[data-testid^="kill-"]').first();
    await expect(killButton).toBeVisible({ timeout: 30_000 });

    // Type into the xterm terminal and expect `cat` to echo it back. The
    // real <textarea> xterm uses for input is positioned off-screen, so we
    // focus the visible terminal frame and type via the keyboard.
    await page.locator(".xterm-frame").click();
    await page.locator(".xterm-helper-textarea").focus();
    await page.keyboard.type("ping-pong");
    await page.keyboard.press("Enter");
    await expect(page.getByTestId("attach-output")).toContainText("ping-pong", {
      timeout: 30_000,
    });

    // Kill the running exec; `cat` should terminate and report an exit.
    await killButton.click();
    await expect(page.getByTestId("attach-exit")).toBeVisible({
      timeout: 30_000,
    });

    await page.goto("/sessions");
    await page.getByTestId(`stop-session-${sessionId}`).click();
    await expect(page.getByTestId("confirm-stop-session")).toBeVisible();
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

  test("user story 6: agents list shows builtins", async ({ page }) => {
    await page.goto("/agents");
    // Builtins (MI300X, MI350X) are seeded on daemon start.
    await expect(page.getByTestId("agent-row-MI350X")).toBeVisible();
    await page.getByTestId("agent-show-MI350X").click();
    await expect(page.getByTestId("agent-detail-MI350X")).toBeVisible();
  });

  test("user story 6b: edit button opens agent editor", async ({ page }) => {
    await page.goto("/agents");
    await page.getByTestId("edit-agent-MI350X").click();
    await expect(page).toHaveURL(/\/agents\/edit\/MI350X$/);
    await expect(page.getByTestId("agent-editor-name")).toHaveText("MI350X");
    await expect(page.getByTestId("agent-editor-save")).toBeVisible();
  });

  test("user story 7: create a topology on the Topologies page", async ({ page }) => {
    await page.goto("/topologies");
    await page.getByTestId("open-topology-create").click();
    const modal = page.getByTestId("topology-create-modal");
    await expect(modal).toBeVisible();
    await page.getByTestId("topology-create-name").fill(TOPOLOGY);
    await page.getByTestId("topology-create-nodes").fill("1");
    await page.getByTestId("topology-create-gpus").fill("2");
    await page.getByTestId("topology-create-submit").click();
    await expect(modal).toHaveCount(0);
    await expect(page.getByTestId(`topology-row-${TOPOLOGY}`)).toBeVisible();
    await dismissAllToasts(page);
  });

  test("user story 8: pick an existing topology in the profile wizard", async ({ page }) => {
    await page.goto("/profiles");
    await page.getByTestId("open-profile-wizard").click();
    const wizard = page.getByTestId("profile-wizard");
    await expect(wizard).toBeVisible();
    await page.getByTestId("wizard-name").fill(PROFILE_WITH_TOPO);
    await page.getByTestId("wizard-emulator").click();
    await page.getByTestId("wizard-emulator-option-noop").click();
    await page.getByTestId("wizard-topology-mode-existing").click();
    await page.getByTestId("wizard-topology-pick").click();
    await page.getByTestId(`wizard-topology-pick-option-${TOPOLOGY}`).click();
    await page.getByTestId("wizard-submit").click();
    await expect(wizard).toHaveCount(0);
    await expect(page.getByTestId(`profile-row-${PROFILE_WITH_TOPO}`)).toBeVisible();
    await dismissAllToasts(page);
  });

  test("user story 9: delete the test profile and topology", async ({ page }) => {
    await page.goto("/profiles");
    await page.getByTestId(`delete-profile-${PROFILE_WITH_TOPO}`).click();
    await page.getByTestId("confirm-delete-profile-confirm").click();
    await expect(
      page.getByTestId(`profile-row-${PROFILE_WITH_TOPO}`),
    ).toHaveCount(0);

    await page.goto("/topologies");
    await page.getByTestId(`delete-topology-${TOPOLOGY}`).click();
    const confirm = page.getByTestId("confirm-delete-topology");
    await expect(confirm).toBeVisible();
    await page.getByTestId("confirm-delete-topology-confirm").click();
    await expect(page.getByTestId(`topology-row-${TOPOLOGY}`)).toHaveCount(0);
    await dismissAllToasts(page);
  });
});
