import { useEffect, useRef, useState, type ReactNode } from "react";

export interface DropdownOption {
  value: string;
  label: string;
  description?: string;
  badge?: { text: string; tone: "ok" | "warn" | "muted" };
  disabled?: boolean;
}

export interface DropdownProps {
  value: string;
  options: DropdownOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  ariaLabel?: string;
  testId?: string;
  disabled?: boolean;
  /** Renders a hidden native <select> with `name` so form data / tests
   *  can interact with it directly via `selectOption`. */
  name?: string;
}

/**
 * Polished, accessible single-select dropdown.
 *
 * The rendered surface is a custom button + popover, but a real
 * `<select>` is mounted alongside (visually hidden) so that:
 *   - existing Playwright `selectOption` calls keep working,
 *   - the control still submits inside forms,
 *   - assistive tech without JS sees a native control.
 */
export function Dropdown(props: DropdownProps) {
  const {
    value,
    options,
    onChange,
    placeholder = "Select…",
    ariaLabel,
    testId,
    disabled,
    name,
  } = props;
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      if (!wrapRef.current?.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const selected = options.find((o) => o.value === value);

  return (
    <div className="dropdown" ref={wrapRef}>
      <button
        type="button"
        className="dropdown-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        data-testid={testId}
        disabled={disabled}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="dropdown-trigger-label">
          {selected ? selected.label : <span className="dropdown-placeholder">{placeholder}</span>}
        </span>
        {selected?.badge && (
          <span className={`pill pill-${selected.badge.tone}`}>{selected.badge.text}</span>
        )}
        <span className="dropdown-caret" aria-hidden>▾</span>
      </button>
      {open && (
        <ul className="dropdown-menu" role="listbox" data-testid={testId ? `${testId}-menu` : undefined}>
          {options.length === 0 && <li className="dropdown-empty">No options</li>}
          {options.map((opt) => (
            <li
              key={opt.value}
              role="option"
              aria-selected={opt.value === value}
              aria-disabled={opt.disabled}
              data-testid={testId ? `${testId}-option-${opt.value}` : undefined}
              className={`dropdown-option${opt.value === value ? " is-selected" : ""}${opt.disabled ? " is-disabled" : ""}`}
              onClick={() => {
                if (opt.disabled) return;
                onChange(opt.value);
                setOpen(false);
              }}
            >
              <div className="dropdown-option-main">
                <span className="dropdown-option-label">{opt.label}</span>
                {opt.badge && (
                  <span className={`pill pill-${opt.badge.tone}`}>{opt.badge.text}</span>
                )}
              </div>
              {opt.description && (
                <span className="dropdown-option-desc">{opt.description}</span>
              )}
            </li>
          ))}
        </ul>
      )}
      {name && (
        <select
          aria-hidden
          tabIndex={-1}
          className="dropdown-native"
          name={name}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          data-testid={testId ? `${testId}-native` : undefined}
        >
          {options.map((o) => (
            <option key={o.value} value={o.value}>{o.label}</option>
          ))}
        </select>
      )}
    </div>
  );
}

export interface MultiSelectProps {
  values: string[];
  options: { value: string; label: string; description?: string }[];
  onChange: (values: string[]) => void;
  emptyHint?: ReactNode;
  testId?: string;
}

export function MultiSelect(props: MultiSelectProps) {
  const { values, options, onChange, emptyHint, testId } = props;
  if (options.length === 0) {
    return (
      <div className="multiselect multiselect-empty" data-testid={testId}>
        {emptyHint ?? <span className="muted">No options available</span>}
      </div>
    );
  }
  function toggle(v: string) {
    onChange(values.includes(v) ? values.filter((x) => x !== v) : [...values, v]);
  }
  return (
    <div className="multiselect" data-testid={testId}>
      {options.map((o) => {
        const on = values.includes(o.value);
        return (
          <label
            key={o.value}
            className={`multiselect-chip${on ? " is-on" : ""}`}
            data-testid={testId ? `${testId}-${o.value}` : undefined}
          >
            <input
              type="checkbox"
              checked={on}
              onChange={() => toggle(o.value)}
            />
            <span>{o.label}</span>
            {o.description && <span className="muted">{o.description}</span>}
          </label>
        );
      })}
    </div>
  );
}
