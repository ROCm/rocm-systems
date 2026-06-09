import { ClassicPreset } from "rete";

/** Hardware component output — feeds into parent's children input or connector's from/to */
export const componentSocket = new ClassicPreset.Socket("Component");

/** Socket carrying a hardware socket reference (port name + protocol) */
export const hwSocket = new ClassicPreset.Socket("HwSocket");

/** Expression/lambda — output of math ops, input to connector mask */
export const exprSocket = new ClassicPreset.Socket("Expression");

/** Numeric constant value */
export const numberSocket = new ClassicPreset.Socket("Number");

/** Config key-value pair */
export const configSocket = new ClassicPreset.Socket("Config");
