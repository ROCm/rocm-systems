# Mirage Dashboard – User Stories

The dashboard is the visual control plane for `mirage`. The goal is a
desktop-class experience inspired by **Docker Desktop** (resource cards,
prominent status, persistent left rail, command bar) and the **iOS
Simulator** (single hero device card, scoped device-detail toolbar, plays
of subtle motion when state changes).

Every story has acceptance criteria. The `e2e/dashboard.spec.ts` Playwright
suite covers each one end-to-end via the same daemon binary the user runs.

## Personas

- **DevOps Diego** — runs a long-lived `mirage daemon` and uses the
  dashboard to inspect profiles and clean up stale sessions.
- **Researcher Rita** — wants to launch a one-off session, attach to a
  command, see its output stream, and tear it down.
- **Plugin author Pat** — needs to see what emulator backends are
  installed on this host and which plugins they expose.

## 1. Overview at a glance (Diego)

**As a** dashboard user **I want** the first screen to show me how many
profiles and sessions exist, the health of each running session, and
where on disk mirage stores its data **so that** I can confirm the
daemon is healthy in under three seconds.

### Acceptance criteria

- The route `/` renders a 4-card metric strip: `Profiles`, `Sessions`,
  `Healthy`, `Execs running` (last number aggregates across sessions).
- Every metric card is keyboard-focusable and links to its detail page.
- A `Storage` panel lists the five XDG-derived paths in a copy-friendly
  table.
- A `System` panel shows the daemon version + the list of detected
  emulator backends (each with an `installed` / `not installed` badge).
- The page polls `/api/sessions` every 3 s and updates without a full
  reload.

## 2. Browse and create profiles (Pat)

**As a** plugin author **I want** to pick the emulator backend from a
dropdown (showing which are installed locally) and choose plugins from
a multi-select **so that** I never have to remember which emulator
strings the daemon accepts.

### Acceptance criteria

- `/profiles` lists every profile in a sortable table with: name, emulator
  badge, nodes, GPUs/node, exec mode, description.
- A primary `New profile…` button opens a modal with:
  - Name field (required, lowercased automatically).
  - Emulator dropdown populated from `/api/emulators` (shows installed
    badge per entry; defaults to the daemon's `default_emulator`).
  - Nodes / GPUs per node number steppers.
  - Exec mode segmented control (`Functional` / `Clocked`).
  - Plugins multi-select fed by the chosen emulator's
    `available_plugins` list.
  - Description textarea.
- The modal validates the name, disables submit while inflight, surfaces
  daemon errors inline, and closes on success while triggering a table
  refresh.
- Each profile row exposes a kebab menu with `Duplicate`, `Edit JSON`,
  `Delete` (Delete asks for confirmation in a modal, not a `confirm()`).

## 3. Launch a session and watch it boot (Rita)

**As a** researcher **I want** to start a session against a profile and
watch the health badge transition from `pending` → `starting` → `ready`
**so that** I know exactly when I can attach an exec.

### Acceptance criteria

- `/sessions` shows every session as a card (iOS-simulator-style),
  not just a row: profile name, ID, age, health badge, exec count.
- Cards expose `Open`, `Stop`, and `Copy ID`.
- A persistent toolbar carries `New session…`, a profile filter
  dropdown, and a `Refresh` button.
- Starting a session opens a modal pre-filled with the most-recently
  used profile. While the session is starting the card shows an animated
  badge and a progress shimmer.
- The status badge derives from `health.state` plus `health.healthy`
  (`pending`/`starting`/`ready`/`unhealthy`/`terminal`).

## 4. Drive an exec with live output (Rita)

**As a** researcher **I want** to type a shell command, hit run, and
see streamed output in a real terminal pane (xterm) **so that** I can
iterate on a debug command without leaving the browser.

### Acceptance criteria

- `/sessions/:id` shows session metadata in a hero card, then a
  two-column layout: exec list on the left, attached terminal on the
  right.
- The terminal is an `xterm.js` instance bound to the daemon
  `/attach` WebSocket. Resize the column → terminal resizes.
- A command palette at the bottom: command input, `+` env-var chip
  input (KEY=VALUE), `keep` checkbox, `Run` button. Submitting selects
  the new exec automatically.
- Each exec row shows: id, started timestamp, runtime, exit code (red
  if non-zero, green if 0, neutral while running), kebab menu (`Attach`,
  `Send SIGINT`, `Send SIGKILL`, `Remove`).
- The kebab is keyboard-navigable.

## 5. Trust the dashboard's status (Diego)

**As an** operator **I want** transient daemon failures (network blip,
500 from the API) to show as a toast that I can dismiss **so that** I
don't lose state when one request fails.

### Acceptance criteria

- Every page registers a global toast handler. API errors push a toast
  with the HTTP status + message; toasts auto-dismiss after 6 s.
- Toasts stack bottom-right, are keyboard-dismissable (`Esc`).
- A success toast appears after profile create, session create, session
  destroy, and exec remove.

## 6. Keyboard-first navigation (everyone)

- `g o`, `g p`, `g s` navigate to Overview / Profiles / Sessions.
- `Cmd/Ctrl-K` opens a command palette (out of scope for this PR, but
  the layout reserves the slot in the header).
- Sidebar links are reachable via `Tab` and respect `:focus-visible`.
