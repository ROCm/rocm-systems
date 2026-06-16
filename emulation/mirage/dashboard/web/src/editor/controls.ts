import { ClassicPreset } from "rete";

/** Text input control shown inside a node. */
export class TextControl extends ClassicPreset.Control {
  public label?: string;

  constructor(
    public value: string,
    public onChange?: (v: string) => void,
    label?: string,
  ) {
    super();
    this.label = label;
  }

  setValue(v: string) {
    this.value = v;
    this.onChange?.(v);
  }
}

/** Numeric input control shown inside a node. */
export class NumberControl extends ClassicPreset.Control {
  public label?: string;

  constructor(
    public value: number,
    public onChange?: (v: number) => void,
    label?: string,
  ) {
    super();
    this.label = label;
  }

  setValue(v: number) {
    this.value = v;
    this.onChange?.(v);
  }
}

/** Dropdown / select control. */
export class SelectControl extends ClassicPreset.Control {
  public label?: string;

  constructor(
    public value: string,
    public options: string[],
    public onChange?: (v: string) => void,
    label?: string,
  ) {
    super();
    this.label = label;
  }

  setValue(v: string) {
    this.value = v;
    this.onChange?.(v);
  }
}
