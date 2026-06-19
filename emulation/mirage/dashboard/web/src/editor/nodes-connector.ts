import { ClassicPreset } from "rete";
import { componentSocket, hwSocket, exprSocket } from "./sockets";
import { NumberControl } from "./controls";

// ─── Connector Node ───────────────────────────────────────────────────────────
// Represents a link pattern between hardware sockets.
// Inputs:  from (hwSocket), to (hwSocket), mask (exprSocket)
// Output:  plugs into a HardwareNode's "links" input

export class ConnectorNode extends ClassicPreset.Node {
  width = 260;
  height = 260;

  constructor(label?: string) {
    super(label ?? "Connector");

    // The "from" socket reference — connect a HardwareNode's output socket here
    this.addInput("from", new ClassicPreset.Input(hwSocket, "▶ from"));

    // The "to" socket reference — connect a HardwareNode's output socket here
    this.addInput("to", new ClassicPreset.Input(hwSocket, "◀ to"));

    // Mask expression — connect a math/expression node's output here
    this.addInput("mask", new ClassicPreset.Input(exprSocket, "mask (expr)"));

    // Latency & weight controls
    this.addControl("latency", new NumberControl(1, undefined, "latency"));
    this.addControl("weight", new NumberControl(1, undefined, "weight"));

    // Output connects to a HardwareNode's "links" input
    this.addOutput("link", new ClassicPreset.Output(componentSocket, "→ links"));
  }
}
