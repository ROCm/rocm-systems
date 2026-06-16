import { ClassicPreset } from "rete";
import { componentSocket, hwSocket } from "./sockets";
import { TextControl, NumberControl } from "./controls";

// ─── Component type metadata ──────────────────────────────────────────────────

export interface SocketDef {
  name: string;
  direction: "in" | "out";
  protocol: string;
}

export interface ConfigDef {
  key: string;
  defaultValue: string;
}

export interface ComponentTypeDef {
  label: string;
  type: string;
  sockets: SocketDef[];
  configs: ConfigDef[];
  canHaveChildren: boolean;
}

export const COMPONENT_TYPES: ComponentTypeDef[] = [
  {
    label: "SOC",
    type: "soc",
    sockets: [],
    configs: [{ key: "arch", defaultValue: "cdna4" }],
    canHaveChildren: true,
  },
  {
    label: "GPU Memory",
    type: "gpu_memory",
    sockets: [{ name: "cpl", direction: "in", protocol: "memory" }],
    configs: [],
    canHaveChildren: false,
  },
  {
    label: "IOD",
    type: "iod",
    sockets: [
      { name: "msc.cpl_[n]", direction: "in", protocol: "memory" },
      { name: "peer_cpl", direction: "in", protocol: "memory" },
      { name: "peer_req", direction: "out", protocol: "memory" },
    ],
    configs: [{ key: "num_hbm_stacks", defaultValue: "4" }],
    canHaveChildren: false,
  },
  {
    label: "XCD",
    type: "xcd",
    sockets: [{ name: "l2.req", direction: "out", protocol: "memory" }],
    configs: [],
    canHaveChildren: true,
  },
  {
    label: "L2 Cache",
    type: "l2_cache",
    sockets: [
      { name: "cpl_[n]", direction: "in", protocol: "memory" },
      { name: "req", direction: "out", protocol: "memory" },
    ],
    configs: [],
    canHaveChildren: false,
  },
  {
    label: "Command Processor",
    type: "command_processor",
    sockets: [{ name: "req_[n]", direction: "out", protocol: "dispatch" }],
    configs: [{ key: "max_queued_packets", defaultValue: "0" }],
    canHaveChildren: false,
  },
  {
    label: "Shader Engine",
    type: "shader_engine",
    sockets: [],
    configs: [],
    canHaveChildren: true,
  },
  {
    label: "Compute Unit",
    type: "compute_unit",
    sockets: [
      { name: "cpl", direction: "in", protocol: "dispatch" },
      { name: "req", direction: "out", protocol: "memory" },
      { name: "adj_cpl", direction: "in", protocol: "untyped" },
      { name: "adj_req", direction: "out", protocol: "untyped" },
      { name: "adj_cpl_r", direction: "in", protocol: "untyped" },
      { name: "adj_req_r", direction: "out", protocol: "untyped" },
    ],
    configs: [
      { key: "num_wf_slots", defaultValue: "10" },
      { key: "sgprs_per_wf", defaultValue: "104" },
      { key: "vgprs_per_wf", defaultValue: "256" },
      { key: "lds_size_kb", defaultValue: "160" },
      { key: "functional_quantum", defaultValue: "0" },
    ],
    canHaveChildren: false,
  },
];

// ─── Hardware Component Node ──────────────────────────────────────────────────

export class HardwareNode extends ClassicPreset.Node {
  width = 220;
  height = 0; // computed after adding I/O

  // Metadata for serialization
  componentType: string;
  instanceName: string;

  constructor(typeDef: ComponentTypeDef, instanceName?: string) {
    super(typeDef.label);
    this.componentType = typeDef.type;
    this.instanceName = instanceName ?? typeDef.type;

    // Name control
    const nameCtrl = new TextControl(this.instanceName, (v) => {
      this.instanceName = v;
    }, "name");
    this.addControl("name", nameCtrl);

    // Count control (how many instances, e.g. "8" for xcd[0:8])
    const countCtrl = new NumberControl(1, undefined, "count");
    this.addControl("count", countCtrl);

    // Children input — other HardwareNodes connect here
    if (typeDef.canHaveChildren) {
      this.addInput("children", new ClassicPreset.Input(componentSocket, "children", true));
    }

    // Links input — ConnectorNodes plug in here
    this.addInput("links", new ClassicPreset.Input(componentSocket, "links", true));

    // Parent output — connects to another HardwareNode's children input
    this.addOutput("parent", new ClassicPreset.Output(componentSocket, "parent"));

    // Hardware sockets as outputs (from/to references for connectors)
    for (const s of typeDef.sockets) {
      const dir = s.direction === "in" ? "◀" : "▶";
      const label = `${dir} ${s.name} : ${s.protocol}`;
      this.addOutput(`sock_${s.name}`, new ClassicPreset.Output(hwSocket, label));
    }

    // Config controls
    for (const c of typeDef.configs) {
      const ctrl = new TextControl(c.defaultValue, undefined, c.key);
      this.addControl(`cfg_${c.key}`, ctrl);
    }

    this.height = this.computeHeight(typeDef);
  }

  private computeHeight(def: ComponentTypeDef): number {
    const rows =
      2 + // name + count controls
      (def.canHaveChildren ? 1 : 0) +
      1 + // links input
      1 + // parent output
      def.sockets.length +
      def.configs.length;
    return 60 + rows * 34;
  }
}
