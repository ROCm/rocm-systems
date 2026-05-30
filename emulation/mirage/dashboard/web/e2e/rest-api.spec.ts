import { test, expect } from "@playwright/test";

/**
 * REST + WebSocket attach integration tests that talk directly to the
 * daemon's generated axum router. These exercise the `ctl_dsl!` REST
 * surface without going through the dashboard UI so we can assert the
 * exact wire contract the dashboard depends on.
 */

const API = "http://localhost:50051/api";

async function jsonPost<T>(path: string, body: unknown = {}): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}: ${await res.text()}`);
  return res.json();
}

test.describe("REST API", () => {
  test.beforeEach(async ({ request }) => {
    await request.post(`${API}/__reset`);
  });

  test("get_overview returns daemon state", async () => {
    const r = await jsonPost<{
      simulator_count: number;
      profile_count: number;
      session_count: number;
    }>("/get_overview");
    expect(r.simulator_count).toBe(1);
    expect(r.profile_count).toBe(0);
    expect(r.session_count).toBe(0);
  });

  test("list_simulators exposes the builtin rocjitsu", async () => {
    const r = await jsonPost<{
      simulators: { name: string; supported_gpus: { name: string }[] }[];
    }>("/list_simulators");
    expect(r.simulators).toHaveLength(1);
    expect(r.simulators[0].name).toBe("rocjitsu");
    expect(r.simulators[0].supported_gpus.map((g) => g.name)).toContain(
      "MI300X",
    );
  });

  test("create_profile validates the simulator and mode", async () => {
    const bad = await jsonPost<{ ok: boolean; error?: string }>(
      "/create_profile",
      {
        name: "p1",
        simulator: "missing",
        mode: "functional",
        gpu: "MI300X",
        gpus_per_node: 1,
        nodes: 1,
      },
    );
    expect(bad.ok).toBe(false);
    expect(bad.error).toMatch(/simulator/);

    const ok = await jsonPost<{ ok: boolean }>("/create_profile", {
      name: "p1",
      simulator: "rocjitsu",
      mode: "functional",
      gpu: "MI300X",
      gpus_per_node: 1,
      nodes: 1,
    });
    expect(ok.ok).toBe(true);
  });

  test("boot + list_sessions + shutdown roundtrip", async () => {
    await jsonPost("/create_profile", {
      name: "prof",
      simulator: "rocjitsu",
      mode: "functional",
      gpu: "MI300X",
      gpus_per_node: 1,
      nodes: 1,
    });
    const boot = await jsonPost<{ ok: boolean; container_ids: string[] }>(
      "/boot",
      { name: "s1", profile: "prof", image: "ubuntu", volumes: [] },
    );
    expect(boot.ok).toBe(true);
    expect(boot.container_ids.length).toBeGreaterThan(0);

    const list = await jsonPost<{
      sessions: { name: string; profile: string }[];
    }>("/list_sessions");
    expect(list.sessions).toHaveLength(1);
    expect(list.sessions[0].name).toBe("s1");

    const down = await jsonPost<{ ok: boolean }>("/shutdown", { name: "s1" });
    expect(down.ok).toBe(true);
  });

  test("duplicate profile names are rejected", async () => {
    const body = {
      name: "dupe",
      simulator: "rocjitsu",
      mode: "functional",
      gpu: "MI300X",
      gpus_per_node: 1,
      nodes: 1,
    };
    await jsonPost<{ ok: boolean }>("/create_profile", body);
    const again = await jsonPost<{ ok: boolean; error?: string }>(
      "/create_profile",
      body,
    );
    expect(again.ok).toBe(false);
    expect(again.error).toMatch(/already exists/);
  });
});

test.describe("WebSocket attach", () => {
  test.beforeEach(async ({ request }) => {
    await request.post(`${API}/__reset`);
    await jsonPost("/create_profile", {
      name: "attach-prof",
      simulator: "rocjitsu",
      mode: "functional",
      gpu: "MI300X",
      gpus_per_node: 1,
      nodes: 1,
    });
    await jsonPost("/boot", {
      name: "attach-sess",
      profile: "attach-prof",
      image: "ubuntu",
      volumes: [],
    });
  });

  test("exec returns an exec id and attach websocket delivers a reply frame", async ({
    page,
  }) => {
    const exec = await jsonPost<{ exec_id: string }>("/exec", {
      session_name: "attach-sess",
      command: ["echo", "hello"],
    });
    expect(exec.exec_id).toMatch(/^session\//);

    // Drive the WebSocket from inside the browser context so CORS is not
    // an issue and the environment matches what the dashboard sees.
    await page.goto("/");
    const frames = await page.evaluate(async (execId: string) => {
      const url = `ws://localhost:50051/api/attach?exec_id=${encodeURIComponent(execId)}`;
      return await new Promise<string[]>((resolve) => {
        const collected: string[] = [];
        const ws = new WebSocket(url);
        const giveUp = setTimeout(() => resolve(collected), 5000);
        ws.onmessage = (ev) => collected.push(ev.data as string);
        ws.onclose = () => {
          clearTimeout(giveUp);
          resolve(collected);
        };
        ws.onerror = () => {
          clearTimeout(giveUp);
          resolve(collected);
        };
      });
    }, exec.exec_id);

    // Expect at least an output frame followed by a reply frame.
    expect(frames.length).toBeGreaterThanOrEqual(2);
    const parsed = frames.map((f) => JSON.parse(f));
    const types = parsed.map((m) => m.type);
    expect(types).toContain("output");
    expect(types[types.length - 1]).toBe("reply");
    expect(parsed[parsed.length - 1].data.exit_code).toBe(0);
  });
});
