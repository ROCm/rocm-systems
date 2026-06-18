import { ClassicPreset } from "rete";
import { exprSocket, numberSocket } from "./sockets";
import { NumberControl, TextControl } from "./controls";

// ─── Variable Reference ───────────────────────────────────────────────────────
// Outputs a named loop variable (i, j, k) as an expression.

export class VarNode extends ClassicPreset.Node {
  width = 160;
  height = 130;

  constructor(name = "i") {
    super("Var");
    this.addControl("varName", new TextControl(name, undefined, "var"));
    this.addOutput("out", new ClassicPreset.Output(exprSocket, "expr"));
  }
}

// ─── Numeric Constant ─────────────────────────────────────────────────────────

export class NumberNode extends ClassicPreset.Node {
  width = 160;
  height = 130;

  constructor(value = 0) {
    super("Number");
    this.addControl("value", new NumberControl(value, undefined, "value"));
    this.addOutput("out", new ClassicPreset.Output(numberSocket, "value"));
    // Also output as expression for use in masks
    this.addOutput("expr", new ClassicPreset.Output(exprSocket, "expr"));
  }
}

// ─── Binary Math Operation ────────────────────────────────────────────────────
// add, subtract, multiply, divide, mod

export type MathOp = "add" | "sub" | "mul" | "div" | "mod";

const MATH_OP_SYMBOLS: Record<MathOp, string> = {
  add: "+",
  sub: "−",
  mul: "×",
  div: "÷",
  mod: "%",
};

export class MathNode extends ClassicPreset.Node {
  width = 180;
  height = 180;
  op: MathOp;

  constructor(op: MathOp = "add") {
    super(`Math (${MATH_OP_SYMBOLS[op]})`);
    this.op = op;
    this.addInput("a", new ClassicPreset.Input(exprSocket, "a"));
    this.addInput("b", new ClassicPreset.Input(exprSocket, "b"));
    this.addOutput("out", new ClassicPreset.Output(exprSocket, "expr"));
  }
}

// ─── Comparison Operation ─────────────────────────────────────────────────────
// eq, ne, gt, lt, gte, lte — produce a boolean expression (lambda)

export type CmpOp = "eq" | "ne" | "gt" | "lt" | "gte" | "lte";

const CMP_OP_SYMBOLS: Record<CmpOp, string> = {
  eq: "==",
  ne: "!=",
  gt: ">",
  lt: "<",
  gte: ">=",
  lte: "<=",
};

export class CompareNode extends ClassicPreset.Node {
  width = 180;
  height = 180;
  op: CmpOp;

  constructor(op: CmpOp = "eq") {
    super(`Compare (${CMP_OP_SYMBOLS[op]})`);
    this.op = op;
    this.addInput("a", new ClassicPreset.Input(exprSocket, "a"));
    this.addInput("b", new ClassicPreset.Input(exprSocket, "b"));
    this.addOutput("out", new ClassicPreset.Output(exprSocket, "expr"));
  }
}

// ─── Logical Operation ────────────────────────────────────────────────────────
// and, or, not

export type LogicOp = "and" | "or";

export class LogicNode extends ClassicPreset.Node {
  width = 180;
  height = 180;
  op: LogicOp;

  constructor(op: LogicOp = "and") {
    super(`Logic (${op.toUpperCase()})`);
    this.op = op;
    this.addInput("a", new ClassicPreset.Input(exprSocket, "a"));
    this.addInput("b", new ClassicPreset.Input(exprSocket, "b"));
    this.addOutput("out", new ClassicPreset.Output(exprSocket, "expr"));
  }
}

export class NotNode extends ClassicPreset.Node {
  width = 160;
  height = 150;

  constructor() {
    super("Logic (NOT)");
    this.addInput("a", new ClassicPreset.Input(exprSocket, "a"));
    this.addOutput("out", new ClassicPreset.Output(exprSocket, "expr"));
  }
}

// ─── Mask Builtins ────────────────────────────────────────────────────────────
// cross, cross_no_self, pairs — convenience nodes that expand to expressions

export type MaskBuiltin = "cross" | "cross_no_self" | "pairs";

export class MaskBuiltinNode extends ClassicPreset.Node {
  width = 200;
  height = 180;
  builtin: MaskBuiltin;

  constructor(builtin: MaskBuiltin = "cross") {
    super(`Mask: ${builtin}`);
    this.builtin = builtin;

    if (builtin === "pairs") {
      // pairs(k, k+1): takes a variable and offset
      this.addInput("var", new ClassicPreset.Input(exprSocket, "variable"));
      this.addControl("offset", new NumberControl(1, undefined, "offset"));
    } else {
      // cross / cross_no_self: takes two range variables
      this.addInput("varA", new ClassicPreset.Input(exprSocket, "var A"));
      this.addInput("varB", new ClassicPreset.Input(exprSocket, "var B"));
    }

    this.addOutput("out", new ClassicPreset.Output(exprSocket, "mask expr"));
  }
}

// ─── Range Node ───────────────────────────────────────────────────────────────
// ─── (RangeNode removed — for_ranges are derived from component counts) ─────
