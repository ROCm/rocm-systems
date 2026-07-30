import { NodeEditor, type GetSchemes, ClassicPreset } from "rete";
import { AreaPlugin, AreaExtensions } from "rete-area-plugin";
import { ConnectionPlugin, Presets as ConnectionPresets } from "rete-connection-plugin";
import { ReactPlugin, Presets, type ReactArea2D } from "rete-react-plugin";
import { AutoArrangePlugin, Presets as ArrangePresets } from "rete-auto-arrange-plugin";
import { ContextMenuPlugin, Presets as ContextMenuPresets, type ContextMenuExtra } from "rete-context-menu-plugin";
import { HistoryPlugin, Presets as HistoryPresets } from "rete-history-plugin";
import { ReroutePlugin, RerouteExtensions, type RerouteExtra } from "rete-connection-reroute-plugin";
import { createRoot } from "react-dom/client";

import { HardwareNode, COMPONENT_TYPES } from "./nodes-hardware";
import { ConnectorNode } from "./nodes-connector";
import {
  VarNode, NumberNode,
  MathNode, type MathOp,
  CompareNode, type CmpOp,
  LogicNode,
  NotNode,
  MaskBuiltinNode,
} from "./nodes-math";
import { TextControl, NumberControl, SelectControl } from "./controls";
import {
  TextControlComponent,
  NumberControlComponent,
  SelectControlComponent,
} from "./control-components";
import { ColoredNode } from "./custom-node";

// ─── Type plumbing ────────────────────────────────────────────────────────────

type AnyNode =
  | HardwareNode
  | ConnectorNode
  | VarNode
  | NumberNode
  | MathNode
  | CompareNode
  | LogicNode
  | NotNode
  | MaskBuiltinNode;

type Conn = ClassicPreset.Connection<AnyNode, AnyNode> & { isLoop?: boolean };
type Schemes = GetSchemes<AnyNode, Conn>;
type AreaExtra = ReactArea2D<Schemes> | ContextMenuExtra | RerouteExtra;

export type TopologyEditor = {
  editor: NodeEditor<Schemes>;
  area: AreaPlugin<Schemes, AreaExtra>;
  arrange: () => Promise<void>;
  destroy: () => void;
};

// ─── Editor factory ───────────────────────────────────────────────────────────

export async function createEditor(container: HTMLElement): Promise<TopologyEditor> {
  const editor = new NodeEditor<Schemes>();
  const area = new AreaPlugin<Schemes, AreaExtra>(container);
  const connection = new ConnectionPlugin<Schemes, AreaExtra>();
  const render = new ReactPlugin<Schemes, AreaExtra>({ createRoot });
  const arrange = new AutoArrangePlugin<Schemes>();
  const history = new HistoryPlugin<Schemes>();
  const reroute = new ReroutePlugin<Schemes>();

  // ── Context menu ──────────────────────────────────────────────────────────

  const contextMenu = new ContextMenuPlugin<Schemes>({
    items: ContextMenuPresets.classic.setup([
      // Hardware components
      ["Hardware", COMPONENT_TYPES.map((t) => [
        t.label,
        () => new HardwareNode(t),
      ] as [string, () => AnyNode])],
      // Connector
      ["Links", [
        ["Connector", () => new ConnectorNode()] as [string, () => AnyNode],
      ]],
      // Variables & constants
      ["Expression", [
        ["Variable", () => new VarNode()] as [string, () => AnyNode],
        ["Number", () => new NumberNode()] as [string, () => AnyNode],
      ]],
      // Math ops
      ["Math", ([
        ["add", "+"],
        ["sub", "−"],
        ["mul", "×"],
        ["div", "÷"],
        ["mod", "%"],
      ] as [MathOp, string][]).map(
        ([op, sym]) =>
          [`${sym} ${op}`, () => new MathNode(op)] as [string, () => AnyNode],
      )],
      // Comparison ops
      ["Compare", ([
        ["eq", "=="],
        ["ne", "!="],
        ["gt", ">"],
        ["lt", "<"],
        ["gte", ">="],
        ["lte", "<="],
      ] as [CmpOp, string][]).map(
        ([op, sym]) =>
          [`${sym} ${op}`, () => new CompareNode(op)] as [string, () => AnyNode],
      )],
      // Logic ops
      ["Logic", [
        ["AND", () => new LogicNode("and")] as [string, () => AnyNode],
        ["OR", () => new LogicNode("or")] as [string, () => AnyNode],
        ["NOT", () => new NotNode()] as [string, () => AnyNode],
      ]],
      // Mask builtins
      ["Mask", [
        ["cross", () => new MaskBuiltinNode("cross")] as [string, () => AnyNode],
        ["cross_no_self", () => new MaskBuiltinNode("cross_no_self")] as [string, () => AnyNode],
        ["pairs", () => new MaskBuiltinNode("pairs")] as [string, () => AnyNode],
      ]],
    ]),
  });

  // ── Plugin wiring ─────────────────────────────────────────────────────────

  connection.addPreset(ConnectionPresets.classic.setup());

  render.addPreset(Presets.contextMenu.setup());
  render.addPreset(
    Presets.classic.setup({
      customize: {
        node() {
          return ColoredNode as any;
        },
        control(data) {
          if (data.payload instanceof TextControl) return TextControlComponent as any;
          if (data.payload instanceof NumberControl) return NumberControlComponent as any;
          if (data.payload instanceof SelectControl) return SelectControlComponent as any;
          return null;
        },
      },
    }),
  );
  render.addPreset(Presets.reroute.setup({ contextMenu(id) { reroute.remove(id); } }));

  arrange.addPreset(ArrangePresets.classic.setup());
  history.addPreset(HistoryPresets.classic.setup());

  // ── Use plugins ───────────────────────────────────────────────────────────

  editor.addPipe((ctx) => {
    // Remove existing connection for single-connection inputs
    if (ctx.type === "connectioncreate") {
      const { data } = ctx;
      const targetNode = editor.getNode(data.target);
      if (targetNode) {
        const inputKey = data.targetInput;
        const input = targetNode.inputs[inputKey];
        if (input && !input.multipleConnections) {
          const existing = editor
            .getConnections()
            .filter((c) => c.target === data.target && c.targetInput === inputKey);
          for (const c of existing) {
            editor.removeConnection(c.id);
          }
        }
      }
    }
    return ctx;
  });

  // editor→area must be established first so child plugins can find their parent scope
  editor.use(area);

  area.use(connection);
  area.use(render);
  area.use(arrange);
  area.use(contextMenu);
  area.use(history);
  render.use(reroute as any);

  const selector = AreaExtensions.selector();
  const accumulating = AreaExtensions.accumulateOnCtrl();

  AreaExtensions.simpleNodesOrder(area);
  AreaExtensions.selectableNodes(area, selector, { accumulating });
  RerouteExtensions.selectablePins(reroute, selector, accumulating);

  // ── Auto-arrange helper ───────────────────────────────────────────────────

  async function doArrange() {
    await arrange.layout();
    await AreaExtensions.zoomAt(area, editor.getNodes());
  }

  return {
    editor,
    area,
    arrange: doArrange,
    destroy() {
      area.destroy();
    },
  };
}
