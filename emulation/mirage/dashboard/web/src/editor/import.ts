import { ClassicPreset, type NodeEditor, type GetSchemes } from "rete";
import { HardwareNode, COMPONENT_TYPES, type ComponentTypeDef } from "./nodes-hardware";
import { ConnectorNode } from "./nodes-connector";
import { VarNode, CompareNode } from "./nodes-math";
import type { NumberControl, TextControl } from "./controls";

// ─── JSON config types ────────────────────────────────────────────────────────

interface ConfigEntry {
  key: string;
  value: string;
}

interface ComponentDef {
  name: string;
  type: string;
  config?: ConfigEntry[];
  children?: ComponentDef[];
}

interface ForRange {
  var_name: string;
  start: number;
  end: number;
}

interface LinkDef {
  pattern: string;
  for_ranges?: ForRange[];
  where_expr?: string;
  latency: number;
  weight: number;
}

interface TopologyConfig {
  topology: {
    root: ComponentDef;
    links: LinkDef[];
  };
  [key: string]: unknown;
}

// ─── Type plumbing (mirrors editor.ts) ────────────────────────────────────────

type AnyNode =
  | HardwareNode
  | ConnectorNode
  | VarNode
  | CompareNode;

type Conn = ClassicPreset.Connection<AnyNode, AnyNode> & { isLoop?: boolean };
type Schemes = GetSchemes<AnyNode, Conn>;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function findTypeDef(typeName: string): ComponentTypeDef {
  const td = COMPONENT_TYPES.find((t) => t.type === typeName);
  if (!td) throw new Error(`Unknown component type: ${typeName}`);
  return td;
}

/** Parse "name[start:end]" → { baseName, count } or { baseName, count: 1 }. */
function parseName(raw: string): { baseName: string; count: number } {
  const m = raw.match(/^(.+?)\[(\d+):(\d+)\]$/);
  if (m) {
    return { baseName: m[1], count: Number(m[3]) - Number(m[2]) };
  }
  return { baseName: raw, count: 1 };
}

/** Set a TextControl's value on a node. */
function setTextCtrl(node: AnyNode, key: string, value: string) {
  const ctrl = node.controls[key] as TextControl | undefined;
  if (ctrl) ctrl.setValue(value);
}

/** Set a NumberControl's value on a node. */
function setNumCtrl(node: AnyNode, key: string, value: number) {
  const ctrl = node.controls[key] as NumberControl | undefined;
  if (ctrl) ctrl.setValue(value);
}

// ─── Import ───────────────────────────────────────────────────────────────────

export async function importConfig(
  editor: NodeEditor<Schemes>,
  config: TopologyConfig,
) {
  // Clear existing graph
  for (const c of [...editor.getConnections()]) {
    await editor.removeConnection(c.id);
  }
  for (const n of [...editor.getNodes()]) {
    await editor.removeNode(n.id);
  }

  const { root, links } = config.topology;

  // Map from path string → HardwareNode
  // Store both full paths ("soc.xcd.l2") and root-relative paths ("xcd.l2")
  const pathToNode = new Map<string, HardwareNode>();
  let rootName = "";

  // ── 1. Build the component tree ─────────────────────────────────────────

  async function addComponent(
    def: ComponentDef,
    parentPath: string,
    parentNode: HardwareNode | null,
  ): Promise<HardwareNode> {
    const typeDef = findTypeDef(def.type);
    const { baseName, count } = parseName(def.name);
    const node = new HardwareNode(typeDef, baseName);
    setNumCtrl(node, "count", count);

    // Apply config overrides
    if (def.config) {
      for (const c of def.config) {
        setTextCtrl(node, `cfg_${c.key}`, c.value);
      }
    }

    await editor.addNode(node);

    const fullPath = parentPath ? `${parentPath}.${baseName}` : baseName;
    pathToNode.set(fullPath, node);

    // Also store path relative to root (strips root prefix)
    if (!parentNode) {
      rootName = baseName;
    } else if (fullPath.startsWith(rootName + ".")) {
      pathToNode.set(fullPath.slice(rootName.length + 1), node);
    }

    // Connect to parent
    if (parentNode) {
      const conn = new ClassicPreset.Connection(node, "parent", parentNode, "children");
      await editor.addConnection(conn as Conn);
    }

    // Recurse children
    if (def.children) {
      for (const child of def.children) {
        await addComponent(child, fullPath, node);
      }
    }

    return node;
  }

  await addComponent(root, "", null);

  // ── 2. Build link patterns as ConnectorNodes ────────────────────────────

  for (const link of links) {
    const connector = new ConnectorNode();
    setNumCtrl(connector, "latency", link.latency);
    setNumCtrl(connector, "weight", link.weight);
    await editor.addNode(connector);

    // Parse pattern: "from_path -> to_path"
    const parts = link.pattern.split("->").map((s) => s.trim());
    if (parts.length === 2) {
      const fromSocket = resolveSocketRef(parts[0], pathToNode);
      const toSocket = resolveSocketRef(parts[1], pathToNode);

      if (fromSocket) {
        const conn = new ClassicPreset.Connection(
          fromSocket.node, fromSocket.outputKey, connector, "from",
        );
        await editor.addConnection(conn as Conn);
      }
      if (toSocket) {
        const conn = new ClassicPreset.Connection(
          toSocket.node, toSocket.outputKey, connector, "to",
        );
        await editor.addConnection(conn as Conn);
      }
    }

    // Connect connector to the root SOC's links input
    const socNode = pathToNode.get(root.name.replace(/\[.*\]$/, "").split("[")[0]);
    if (socNode) {
      const conn = new ClassicPreset.Connection(connector, "link", socNode, "links");
      await editor.addConnection(conn as Conn);
    }

    // Build where_expr sub-graph if present
    if (link.where_expr) {
      await buildWhereExpr(editor, connector, link.where_expr, link.for_ranges ?? []);
    }
  }
}

// ─── Socket reference resolution ──────────────────────────────────────────────

/**
 * Given a pattern path like "xcd[i].se[j].cu[k].adj_req" or "iod[i/4].msc.cpl_[i%4]",
 * resolve to the HardwareNode and socket output key.
 * We strip index expressions to find the node path, then find the socket.
 */
function resolveSocketRef(
  ref: string,
  pathToNode: Map<string, HardwareNode>,
): { node: HardwareNode; outputKey: string } | null {
  // Split into segments: "xcd[i].se[j].cu[k].adj_req" → ["xcd[i]", "se[j]", "cu[k]", "adj_req"]
  const segments = ref.split(".");

  // Try different splits: last N segments form the socket name, rest form the node path.
  // Need up to 3 for cases like "iod[i/4].msc.cpl_[i%4]" where socket is "msc.cpl_[n]"
  for (let socketParts = 1; socketParts <= Math.min(3, segments.length - 1); socketParts++) {
    const nodeSegments = segments.slice(0, segments.length - socketParts);
    const socketSegments = segments.slice(segments.length - socketParts);

    // Strip index expressions from node segments to get the base path
    const basePath = nodeSegments.map((s) => s.replace(/\[.*?\]/g, "")).join(".");

    const node = pathToNode.get(basePath);
    if (!node) continue;

    // Rejoin socket segments and normalize for matching
    const socketName = socketSegments.join(".");

    // Match strategy: strip index expressions from both the ref socket name
    // and each output key, comparing base forms
    const refBase = socketName.replace(/\[.*?\]/g, "").replace(/_$/, "");

    for (const key of Object.keys(node.outputs)) {
      const sockDef = key.replace(/^sock_/, "");
      const sockBase = sockDef.replace(/\[.*?\]/g, "").replace(/_$/, "");
      if (sockBase === refBase) return { node, outputKey: key };
    }

    // Also try exact after replacing expressions with [n]
    const exactKey = `sock_${socketName.replace(/\[.*?\]/g, "_[n]")}`;
    if (node.outputs[exactKey]) return { node, outputKey: exactKey };
  }

  return null;
}

// ─── Build where_expr sub-graph ───────────────────────────────────────────────

/**
 * Create VarNode + CompareNode for a where_expr (e.g. "i != j").
 * for_ranges are NOT imported as nodes — they are derived from component counts
 * during export. Only the where_expr filter needs expression nodes.
 */
async function buildWhereExpr(
  editor: NodeEditor<Schemes>,
  connector: ConnectorNode,
  whereExpr: string,
  ranges: ForRange[],
) {
  const cmpNode = parseWhereExpr(whereExpr);
  if (!cmpNode) return;

  await editor.addNode(cmpNode);

  // Find variable names referenced in the expression
  const varNames = ranges.map((r) => r.var_name);
  const tokens = whereExpr.replace(/[()]/g, " ").split(/\s+/).filter(Boolean);
  const referencedVars = tokens.filter((t) => varNames.includes(t));

  // Create VarNodes for each referenced variable and connect to compare inputs
  if (referencedVars.length >= 1) {
    const varA = new VarNode(referencedVars[0]);
    await editor.addNode(varA);
    const connA = new ClassicPreset.Connection(varA, "out", cmpNode, "a");
    await editor.addConnection(connA as Conn);
  }
  if (referencedVars.length >= 2) {
    const varB = new VarNode(referencedVars[1]);
    await editor.addNode(varB);
    const connB = new ClassicPreset.Connection(varB, "out", cmpNode, "b");
    await editor.addConnection(connB as Conn);
  }

  // Connect compare output to connector's mask input
  const maskConn = new ClassicPreset.Connection(cmpNode, "out", connector, "mask");
  await editor.addConnection(maskConn as Conn);
}

/** Parse a simple where_expr string into a CompareNode. */
function parseWhereExpr(expr: string): CompareNode | null {
  const ops: [string, "eq" | "ne" | "gt" | "lt" | "gte" | "lte"][] = [
    ["!=", "ne"],
    ["==", "eq"],
    [">=", "gte"],
    ["<=", "lte"],
    [">", "gt"],
    ["<", "lt"],
  ];

  for (const [sym, op] of ops) {
    if (expr.includes(sym)) {
      return new CompareNode(op);
    }
  }

  return null;
}
