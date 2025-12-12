import { Presets, type ClassicScheme, type RenderEmit } from "rete-react-plugin";
import { HardwareNode } from "./nodes-hardware";
import { ConnectorNode } from "./nodes-connector";
import {
  VarNode, NumberNode, MathNode, CompareNode,
  LogicNode, NotNode, MaskBuiltinNode,
} from "./nodes-math";
import { css } from "styled-components";

const { Node } = Presets.classic;

// ─── Color palette per node category ──────────────────────────────────────────

const NODE_COLORS: Record<string, string> = {
  // Hardware components — warm tones
  soc:               "rgba(220, 80, 60, 0.85)",
  gpu_memory:        "rgba(200, 120, 50, 0.85)",
  iod:               "rgba(180, 100, 70, 0.85)",
  xcd:               "rgba(160, 80, 90, 0.85)",
  l2_cache:          "rgba(140, 90, 140, 0.85)",
  command_processor: "rgba(120, 80, 160, 0.85)",
  shader_engine:     "rgba(90, 100, 180, 0.85)",
  compute_unit:      "rgba(70, 120, 190, 0.85)",

  // Connector — teal
  connector:         "rgba(40, 160, 140, 0.85)",

  // Expression nodes — greens/yellows
  var:               "rgba(80, 160, 80, 0.85)",
  number:            "rgba(100, 150, 60, 0.85)",
  math:              "rgba(140, 140, 50, 0.85)",
  compare:           "rgba(170, 130, 50, 0.85)",
  logic:             "rgba(150, 110, 60, 0.85)",
  not:               "rgba(130, 100, 70, 0.85)",
  mask:              "rgba(100, 130, 100, 0.85)",
};

function getNodeCategory(node: unknown): string {
  if (node instanceof HardwareNode) return node.componentType;
  if (node instanceof ConnectorNode) return "connector";
  if (node instanceof VarNode) return "var";
  if (node instanceof NumberNode) return "number";
  if (node instanceof MathNode) return "math";
  if (node instanceof CompareNode) return "compare";
  if (node instanceof LogicNode) return "logic";
  if (node instanceof NotNode) return "not";
  if (node instanceof MaskBuiltinNode) return "mask";
  return "";
}

function nodeStyles(node: unknown) {
  const category = getNodeCategory(node);
  const color = NODE_COLORS[category];
  if (!color) return () => css``;

  return () => css`
    background: ${color};
  `;
}

// ─── Custom Node Component ────────────────────────────────────────────────────

type Props<S extends ClassicScheme> = {
  data: S["Node"] & { width?: number; height?: number };
  emit: RenderEmit<S>;
};

export function ColoredNode<S extends ClassicScheme>(props: Props<S>) {
  return <Node {...props} styles={nodeStyles(props.data)} />;
}
