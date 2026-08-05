import { useState, useEffect } from "react";
import type { TextControl, NumberControl, SelectControl } from "./controls";

export function TextControlComponent(props: { data: TextControl }) {
  const [value, setValue] = useState(props.data.value);
  useEffect(() => {
    setValue(props.data.value);
  }, [props.data]);

  return (
    <label className="rete-control-label">
      {props.data.label && <span className="rete-control-label-text">{props.data.label}</span>}
      <input
        className="rete-text-control"
        type="text"
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          props.data.setValue(e.target.value);
        }}
        onPointerDown={(e) => e.stopPropagation()}
      />
    </label>
  );
}

export function NumberControlComponent(props: { data: NumberControl }) {
  const [value, setValue] = useState(props.data.value);
  useEffect(() => {
    setValue(props.data.value);
  }, [props.data]);

  return (
    <label className="rete-control-label">
      {props.data.label && <span className="rete-control-label-text">{props.data.label}</span>}
      <input
        className="rete-number-control"
        type="number"
        value={value}
        onChange={(e) => {
          const num = Number(e.target.value);
          setValue(num);
          props.data.setValue(num);
        }}
        onPointerDown={(e) => e.stopPropagation()}
      />
    </label>
  );
}

export function SelectControlComponent(props: { data: SelectControl }) {
  const [value, setValue] = useState(props.data.value);
  useEffect(() => {
    setValue(props.data.value);
  }, [props.data]);

  return (
    <label className="rete-control-label">
      {props.data.label && <span className="rete-control-label-text">{props.data.label}</span>}
      <select
        className="rete-select-control"
        value={value}
        onChange={(e) => {
          setValue(e.target.value);
          props.data.setValue(e.target.value);
        }}
        onPointerDown={(e) => e.stopPropagation()}
      >
        {props.data.options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </label>
  );
}
