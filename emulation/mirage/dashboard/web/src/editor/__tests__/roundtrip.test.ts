import { describe, it, expect } from "vitest";
import { NodeEditor } from "rete";
import { importConfig } from "../import";
import { exportConfig } from "../export";
import { COMPONENT_TYPES } from "../nodes-hardware";

const cdna4Config = {
  max_ticks: 100000,
  num_threads: 1,
  exec_mode: "functional",
  vm: { arch: "cdna4" },
  topology: {
    root: {
      name: "soc",
      type: "soc",
      children: [
        { name: "vram", type: "gpu_memory" },
        {
          name: "iod[0:2]",
          type: "iod",
          config: [{ key: "num_hbm_stacks", value: "4" }],
        },
        {
          name: "xcd[0:8]",
          type: "xcd",
          children: [
            { name: "l2", type: "l2_cache" },
            { name: "cp", type: "command_processor" },
            {
              name: "se[0:4]",
              type: "shader_engine",
              children: [
                {
                  name: "cu[0:8]",
                  type: "compute_unit",
                  config: [
                    { key: "num_wf_slots", value: "10" },
                    { key: "sgprs_per_wf", value: "104" },
                    { key: "vgprs_per_wf", value: "256" },
                    { key: "lds_size_kb", value: "160" },
                  ],
                },
              ],
            },
          ],
        },
      ],
    },
    links: [
      {
        pattern: "xcd[i].cp.req_[j*8+k] -> xcd[i].se[j].cu[k].cpl",
        for_ranges: [
          { var_name: "i", start: 0, end: 8 },
          { var_name: "j", start: 0, end: 4 },
          { var_name: "k", start: 0, end: 8 },
        ],
        latency: 1,
        weight: 2,
      },
      {
        pattern: "xcd[i].se[j].cu[k].req -> xcd[i].l2.cpl_[j*8+k]",
        for_ranges: [
          { var_name: "i", start: 0, end: 8 },
          { var_name: "j", start: 0, end: 4 },
          { var_name: "k", start: 0, end: 8 },
        ],
        latency: 1,
        weight: 10,
      },
      {
        pattern: "xcd[i].l2.req -> iod[i/4].msc.cpl_[i%4]",
        for_ranges: [{ var_name: "i", start: 0, end: 8 }],
        latency: 1,
        weight: 3,
      },
      {
        pattern: "iod[i].peer_req -> iod[j].peer_cpl",
        for_ranges: [
          { var_name: "i", start: 0, end: 2 },
          { var_name: "j", start: 0, end: 2 },
        ],
        where_expr: "i != j",
        latency: 1,
        weight: 1,
      },
      {
        pattern: "xcd[i].se[j].cu[k].adj_req -> xcd[i].se[j].cu[k+1].adj_cpl",
        for_ranges: [
          { var_name: "i", start: 0, end: 8 },
          { var_name: "j", start: 0, end: 4 },
          { var_name: "k", start: 0, end: 7 },
        ],
        latency: 1,
        weight: 2,
      },
      {
        pattern:
          "xcd[i].se[j].cu[k+1].adj_req_r -> xcd[i].se[j].cu[k].adj_cpl_r",
        for_ranges: [
          { var_name: "i", start: 0, end: 8 },
          { var_name: "j", start: 0, end: 4 },
          { var_name: "k", start: 0, end: 7 },
        ],
        latency: 1,
        weight: 2,
      },
    ],
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Build a map of default config values per component type. */
function getDefaults(typeName: string): Map<string, string> {
  const td = COMPONENT_TYPES.find((t) => t.type === typeName);
  if (!td) return new Map();
  return new Map(td.configs.map((c) => [c.key, c.defaultValue]));
}

/** Recursively sort children arrays by name so ordering doesn't affect comparison. */
function normalizeComponent(c: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { name: c.name, type: c.type };
  const defaults = getDefaults(c.type as string);

  // Only keep configs that differ from the component type's defaults
  if (c.config && (c.config as unknown[]).length > 0) {
    const nonDefault = (c.config as { key: string; value: string }[]).filter(
      (cfg) => defaults.get(cfg.key) !== cfg.value,
    );
    if (nonDefault.length > 0) {
      out.config = nonDefault.sort((a, b) => a.key.localeCompare(b.key));
    }
  }
  if (c.children && (c.children as unknown[]).length > 0) {
    out.children = (c.children as Record<string, unknown>[])
      .map(normalizeComponent)
      .sort((a, b) =>
        (a.name as string).localeCompare(b.name as string),
      );
  }
  return out;
}

/**
 * Normalize a socket reference path for comparison.
 * The export always includes the root prefix (e.g. "soc.xcd.l2.req")
 * while the input JSON omits it (e.g. "xcd[i].l2.req").
 * Also strip index expressions like [i], [j*8+k], [n] and trailing _
 * to get a canonical base form for structural comparison.
 */
function normalizeSocketRef(ref: string, rootName: string): string {
  // Strip root prefix if present
  const prefix = rootName + ".";
  let normalized = ref.startsWith(prefix) ? ref.slice(prefix.length) : ref;
  // Strip all index/expression brackets and underscores before them: foo_[n] → foo, bar[i] → bar
  normalized = normalized.replace(/_?\[.*?\]/g, "");
  return normalized;
}

/** Normalize a link for structural comparison. */
function normalizeLink(
  link: Record<string, unknown>,
  rootName: string,
): Record<string, unknown> {
  const pattern = link.pattern as string | undefined;
  let from = "";
  let to = "";
  if (pattern) {
    const parts = pattern.split("->").map((s) => s.trim());
    from = normalizeSocketRef(parts[0], rootName);
    to = normalizeSocketRef(parts[1], rootName);
  }

  const out: Record<string, unknown> = {
    from,
    to,
    latency: link.latency,
    weight: link.weight,
  };

  if (link.for_ranges && (link.for_ranges as unknown[]).length > 0) {
    out.for_ranges = (link.for_ranges as { var_name: string; start: number; end: number }[])
      .sort((a, b) => a.var_name.localeCompare(b.var_name));
  }
  if (link.where_expr) {
    out.where_expr = (link.where_expr as string).replace(/\s+/g, " ").trim();
  }

  return out;
}

function normalizeLinks(links: Record<string, unknown>[], rootName: string) {
  return links
    .map((l) => normalizeLink(l, rootName))
    .sort((a, b) => {
      const keyA = `${a.from}->${a.to}`;
      const keyB = `${b.from}->${b.to}`;
      return keyA.localeCompare(keyB);
    });
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe("import → export roundtrip", () => {
  it("preserves the component tree structure", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    const inputRoot = normalizeComponent(
      cdna4Config.topology.root as unknown as Record<string, unknown>,
    );
    const outputRoot = normalizeComponent(
      exported.topology.root as unknown as Record<string, unknown>,
    );

    expect(outputRoot).toEqual(inputRoot);
  });

  it("preserves the same number of links", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    expect(exported.topology.links).toHaveLength(
      cdna4Config.topology.links.length,
    );
  });

  it("preserves link latency and weight values", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    const inputLinks = cdna4Config.topology.links.map((l) => ({
      latency: l.latency,
      weight: l.weight,
    }));
    const outputLinks = exported.topology.links.map((l) => ({
      latency: l.latency,
      weight: l.weight,
    }));

    // Sort both by latency then weight for order-independent comparison
    const sort = (arr: { latency: number; weight: number }[]) =>
      [...arr].sort((a, b) => a.weight - b.weight || a.latency - b.latency);

    expect(sort(outputLinks)).toEqual(sort(inputLinks));
  });

  it("preserves link socket endpoints (normalized)", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    const rootName = cdna4Config.topology.root.name;
    const inputLinks = normalizeLinks(
      cdna4Config.topology.links as unknown as Record<string, unknown>[],
      rootName,
    );
    const outputLinks = normalizeLinks(
      exported.topology.links as unknown as Record<string, unknown>[],
      rootName,
    );

    // Compare each link's from/to endpoints
    for (let i = 0; i < inputLinks.length; i++) {
      expect(outputLinks[i].from).toBe(inputLinks[i].from);
      expect(outputLinks[i].to).toBe(inputLinks[i].to);
    }
  });

  it("derives for_ranges from component counts", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    const rootName = cdna4Config.topology.root.name;
    const outputLinks = normalizeLinks(
      exported.topology.links as unknown as Record<string, unknown>[],
      rootName,
    );

    // Ranges are auto-derived from the from/to component hierarchy.
    // Valid range end values should correspond to component counts in the config.
    const componentCounts = new Set([8, 4, 2]); // xcd:8, se:4, cu:8, iod:2

    for (const link of outputLinks) {
      const ranges = (link.for_ranges ?? []) as { var_name: string; start: number; end: number }[];
      expect(ranges.length).toBeGreaterThan(0);
      for (const r of ranges) {
        expect(r.start).toBe(0);
        expect(componentCounts.has(r.end)).toBe(true);
      }
    }

    // The cross-product link (iod→iod) should have two ranges with end=2
    const iodLink = outputLinks.find(
      (l) => l.from === "iod.peer_req" && l.to === "iod.peer_cpl",
    );
    expect(iodLink).toBeDefined();
    const iodRanges = (iodLink!.for_ranges ?? []) as { end: number }[];
    expect(iodRanges).toHaveLength(2);
    expect(iodRanges[0].end).toBe(2);
    expect(iodRanges[1].end).toBe(2);
  });

  it("preserves where_expr on links", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    const rootName = cdna4Config.topology.root.name;
    const inputLinks = normalizeLinks(
      cdna4Config.topology.links as unknown as Record<string, unknown>[],
      rootName,
    );
    const outputLinks = normalizeLinks(
      exported.topology.links as unknown as Record<string, unknown>[],
      rootName,
    );

    for (let i = 0; i < inputLinks.length; i++) {
      if (inputLinks[i].where_expr) {
        expect(outputLinks[i].where_expr).toBe(inputLinks[i].where_expr);
      } else {
        expect(outputLinks[i].where_expr).toBeUndefined();
      }
    }
  });

  it("preserves top-level simulation config", async () => {
    const editor = new NodeEditor();
    await importConfig(editor, cdna4Config as never);
    const exported = exportConfig(editor as never);

    expect(exported.vm.arch).toBe(cdna4Config.vm.arch);
    expect(exported.exec_mode).toBe(cdna4Config.exec_mode);
  });
});
