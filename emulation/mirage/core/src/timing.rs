//! The numbers the emulator's timing plane runs on, and how one run's
//! set of them is assembled.
//!
//! The emulator is handed exactly one architecture config, and the
//! timing block in it is the whole description of how fast the part is.
//! There is no second file, no search path and no model to name: what
//! reaches the backend is a self-contained document, so a run's timing
//! is reproducible from the one artefact it was given and that artefact
//! can be handed to somebody else without them needing anything more.
//! Mirage's job is therefore to *bake* the numbers, not to reference
//! them.
//!
//! # Where the numbers come from
//!
//! Three layers, applied in this order and reported in it:
//!
//! 1. The built-in table on the [`AgentDef`](crate::agent::AgentDef).
//!    It sits beside the device geometry mirage already knows, because
//!    it describes the same device, and every value in it has a stated
//!    provenance (see `mirage_builtin::timing`).
//! 2. Overrides stored in the profile, as
//!    `timing.machine.<key>` emulator options.
//! 3. Overrides from a tuning file named on the command line.
//!
//! Each layer may only *replace* a key the layer below already has.
//! A key nobody has heard of is an error rather than a silent addition:
//! a mistyped parameter that quietly becomes a new one is how a config
//! ends up describing a machine nobody built, and it reads as a
//! measurement afterwards.
//!
//! # Confidentiality
//!
//! A tuning file a user supplies is read and merged; it is never written
//! back into mirage's own tree. The merged result lands in the session's
//! scratch directory under `$XDG_RUNTIME_DIR`, which is outside the
//! repository — see [`crate::paths`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use crate::common::{SimpleMap, SimpleValue};

/// Emulator option that turns the timing plane on for a run.
pub const TIMING_ENABLED_OPTION: &str = "timing";

/// Prefix of the emulator option that overrides one timing number.
///
/// One flat option per key rather than a JSON blob in a single option,
/// so a profile stays a document a person can read and edit: the
/// alternative is a JSON string nested inside JSON, which is only
/// writable by a program.
pub const TIMING_OVERRIDE_PREFIX: &str = "timing.machine.";

/// The one key that is not part of the machine description.
///
/// It is carried in the same table so that it merges, validates and
/// reports through the same path as everything else, and is lifted back
/// out to `timing.clock_mhz` when the block is emitted.
pub const CLOCK_KEY: &str = "clock_mhz";

/// Every number the timing plane reads for one device, as a flat map of
/// dotted keys.
///
/// Flat and dotted rather than nested, because the key space is not a
/// tree: the timing plane reads both `matrix_multiply.macs_per_cycle`
/// and `matrix_multiply.macs_per_cycle.f16`, and no nested object can
/// hold a number and an object under one name.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TimingTable(BTreeMap<String, Number>);

// `serde_json::Number` is `PartialEq` and not `Eq` only because it can
// hold a float, and a float is not reflexive when it is NaN. This one
// cannot be: `Number::from_f64` refuses NaN and infinity, so every
// number that can reach this map compares equal to itself. Asserted
// rather than derived so the documents that embed a table — an
// `AgentDef`, and the `ProfileDef` that carries one — keep the `Eq` the
// rest of mirage compares them with.
impl Eq for TimingTable {}

impl TimingTable {
    /// Build a table from `(key, value)` pairs.
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Number)>) -> Self {
        Self(pairs.into_iter().collect())
    }

    /// How many keys the table names.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether the device carries no timing table at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Whether `key` is part of this device's vocabulary.
    #[must_use]
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    /// One value, or `None` when the table does not name it.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Number> {
        self.0.get(key)
    }

    /// Every key, in sorted order.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.0.keys().map(String::as_str)
    }

    /// Replace the values `overrides` names, refusing every key this
    /// table does not already have.
    ///
    /// # Errors
    ///
    /// Returns the unknown keys, sorted. Nothing is applied when any key
    /// is refused: a half-applied override set is a machine description
    /// that neither layer asked for.
    pub fn overlay(&mut self, overrides: &BTreeMap<String, Number>) -> Result<(), Vec<String>> {
        let unknown: Vec<String> = overrides
            .keys()
            .filter(|key| !self.0.contains_key(*key))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            return Err(unknown);
        }
        for (key, value) in overrides {
            self.0.insert(key.clone(), value.clone());
        }
        Ok(())
    }

    /// The `timing` object to bake into the architecture config.
    ///
    /// # Errors
    ///
    /// Returns a message when the table does not say what clock the part
    /// runs at. Every cycle count in the table is meaningless without
    /// one, so a table missing it would produce a config whose numbers
    /// cannot be converted to time.
    pub fn to_config_block(&self) -> Result<Value, String> {
        let clock = self.0.get(CLOCK_KEY).ok_or_else(|| {
            format!(
                "the timing table names no `{CLOCK_KEY}`, so its cycle counts cannot be \
                 turned into time. An agent's table takes it from the device's \
                 `max_engine_clk_fcompute`."
            )
        })?;
        let machine: serde_json::Map<String, Value> = self
            .0
            .iter()
            .filter(|(key, _)| key.as_str() != CLOCK_KEY)
            .map(|(key, value)| (key.clone(), Value::Number(value.clone())))
            .collect();
        Ok(serde_json::json!({
            "enabled": true,
            "clock_mhz": Value::Number(clock.clone()),
            "machine": Value::Object(machine),
        }))
    }
}

/// Read a tuning document into dotted keys.
///
/// Both spellings are accepted, and mixed freely: a leaf may be named by
/// its dotted path (`{"dram.latency_cycles": 900}`) or by nesting
/// (`{"dram": {"latency_cycles": 900}}`). They are the same key. A
/// document may also wrap the whole set in a `machine` object, which is
/// what the emitted config looks like, so a baked config can be edited
/// and fed straight back in.
///
/// # Errors
///
/// Returns a message naming the offending path when the document is not
/// an object, or when a leaf is not a number. Strings are refused rather
/// than parsed: a quoted number in a tuning file is a typo often enough
/// that accepting it costs more than it saves.
pub fn read_tuning_document(document: &Value) -> Result<BTreeMap<String, Number>, String> {
    let mut out = BTreeMap::new();
    let root = match document {
        Value::Object(map) => match map.get("machine") {
            // Only when `machine` is the whole document, so a part whose
            // table one day names a `machine.*` key of its own is not
            // silently unwrapped.
            Some(Value::Object(inner)) if map.len() == 1 => inner,
            _ => map,
        },
        _ => return Err("a tuning file must be a JSON object of timing keys".to_string()),
    };
    flatten_into(root, "", &mut out)?;
    if out.is_empty() {
        return Err("this tuning file names no timing keys".to_string());
    }
    Ok(out)
}

/// Walk one object level, appending its leaves to `out` under `prefix`.
fn flatten_into(
    map: &serde_json::Map<String, Value>,
    prefix: &str,
    out: &mut BTreeMap<String, Number>,
) -> Result<(), String> {
    for (key, value) in map {
        let path = if prefix.is_empty() {
            key.clone()
        } else {
            format!("{prefix}.{key}")
        };
        match value {
            Value::Number(number) => {
                out.insert(path, number.clone());
            }
            Value::Object(inner) => flatten_into(inner, &path, out)?,
            other => {
                return Err(format!(
                    "`{path}` is {}, and a timing value has to be a number",
                    describe(other)
                ));
            }
        }
    }
    Ok(())
}

/// What a value is, for an error message.
///
/// Only ever reached for a leaf that is not a number and not an object,
/// both of which the walk above handles; they are named anyway so the
/// match stays total and a future JSON variant cannot be silently
/// mislabelled.
fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::String(_) => "a string",
        Value::Array(_) => "a list",
        Value::Number(_) => "a number",
        Value::Object(_) => "an object",
    }
}

/// Whether a profile's emulator options ask for the timing plane.
#[must_use]
pub fn enabled_in_options(options: &SimpleMap) -> bool {
    match options.get(TIMING_ENABLED_OPTION) {
        Some(SimpleValue::Boolean(on)) => *on,
        // A profile is a hand-editable document and JSON has no way to
        // hint that `"true"` was meant as a boolean, so the string is
        // taken at its word rather than read as "not false".
        Some(SimpleValue::String(text)) => text.eq_ignore_ascii_case("true"),
        Some(SimpleValue::Number(n)) => *n != 0,
        None => false,
    }
}

/// The timing overrides a profile carries, as dotted keys.
///
/// # Errors
///
/// Returns a message naming the option when a `timing.machine.*` value
/// is not a number.
pub fn overrides_from_options(options: &SimpleMap) -> Result<BTreeMap<String, Number>, String> {
    let mut out = BTreeMap::new();
    for (option, value) in options {
        let Some(key) = option.strip_prefix(TIMING_OVERRIDE_PREFIX) else {
            continue;
        };
        // Numbers arrive as strings because a `SimpleValue::Number` is an
        // integer and half these keys are rates. Both spellings are read
        // so that a profile written by hand with `"xcds": 8` works.
        let parsed = match value {
            SimpleValue::Number(n) => Some(Number::from(*n)),
            SimpleValue::String(text) => serde_json::from_str::<Number>(text.trim()).ok(),
            SimpleValue::Boolean(_) => None,
        };
        let number = parsed.ok_or_else(|| {
            format!("the profile's `{option}` is not a number, and a timing value has to be one")
        })?;
        out.insert(key.to_string(), number);
    }
    Ok(out)
}

/// Record `enabled` and `overrides` in a profile's emulator options,
/// replacing whatever timing state was there.
///
/// Replacing rather than extending, because the caller has already
/// merged every layer: leaving a stale key behind would put a fourth
/// layer under the three that were reported.
pub fn write_options(options: &mut SimpleMap, enabled: bool, overrides: &BTreeMap<String, Number>) {
    clear_options(options);
    if !enabled {
        return;
    }
    options.insert(
        TIMING_ENABLED_OPTION.to_string(),
        SimpleValue::Boolean(true),
    );
    for (key, value) in overrides {
        options.insert(
            format!("{TIMING_OVERRIDE_PREFIX}{key}"),
            SimpleValue::String(value.to_string()),
        );
    }
}

/// Remove every trace of timing from a profile's emulator options.
///
/// Both halves, or a later run that turns timing back on would pick up
/// overrides for a table nobody looked at in between.
pub fn clear_options(options: &mut SimpleMap) {
    options.remove(TIMING_ENABLED_OPTION);
    options.retain(|key, _| !key.starts_with(TIMING_OVERRIDE_PREFIX));
}

/// Blame a tuning key the device's table does not have.
///
/// The full vocabulary is 80-odd keys, so listing all of it buries the
/// answer. What is printed instead is the keys sharing the misspelling's
/// first path segment, which is where a typo almost always is, and the
/// command that shows the rest.
#[must_use]
pub fn unknown_keys_message(unknown: &[String], table: &TimingTable, source: &str) -> String {
    let mut out = format!(
        "{source} names {} the agent's timing table does not have: {}. \
         A tuning file may only change a number the device already has, never add one — \
         a mistyped key that silently became a new parameter is how a config ends up \
         describing a machine nobody built.",
        if unknown.len() == 1 { "a key" } else { "keys" },
        unknown.join(", ")
    );
    for key in unknown {
        let segment = key.split('.').next().unwrap_or(key);
        let near: Vec<&str> = table
            .keys()
            .filter(|known| known.split('.').next() == Some(segment))
            .collect();
        if !near.is_empty() {
            out.push_str(&format!(
                "\n  under `{segment}` it has: {}",
                near.join(", ")
            ));
        }
    }
    out.push_str("\n`mirage agent show <name>` prints the whole table.");
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn table() -> TimingTable {
        TimingTable::from_pairs([
            (CLOCK_KEY.to_string(), Number::from(2700)),
            ("xcds".to_string(), Number::from(8)),
            ("dram.latency_cycles".to_string(), Number::from(1026)),
            (
                "dram.bytes_per_cycle".to_string(),
                Number::from_f64(2962.963).unwrap(),
            ),
        ])
    }

    #[test]
    fn a_dotted_document_and_a_nested_one_are_the_same_document() {
        let dotted =
            read_tuning_document(&serde_json::json!({"dram.latency_cycles": 900})).unwrap();
        let nested =
            read_tuning_document(&serde_json::json!({"dram": {"latency_cycles": 900}})).unwrap();
        assert_eq!(dotted, nested);
        assert_eq!(dotted["dram.latency_cycles"], Number::from(900));
    }

    /// The emitted config is itself a legal tuning document, so a baked
    /// config can be edited and fed back in with `--timing-tuning`.
    #[test]
    fn a_machine_wrapper_is_unwrapped() {
        let read = read_tuning_document(&serde_json::json!({"machine": {"xcds": 4}})).unwrap();
        assert_eq!(read["xcds"], Number::from(4));
    }

    #[test]
    fn a_non_numeric_leaf_is_refused_by_path() {
        let e = read_tuning_document(&serde_json::json!({"dram": {"latency_cycles": "900"}}))
            .unwrap_err();
        assert!(e.contains("dram.latency_cycles"), "{e}");
    }

    #[test]
    fn an_empty_document_is_refused() {
        assert!(read_tuning_document(&serde_json::json!({})).is_err());
        assert!(read_tuning_document(&serde_json::json!([1, 2])).is_err());
    }

    #[test]
    fn an_overlay_replaces_and_never_adds() {
        let mut t = table();
        let mut over = BTreeMap::new();
        over.insert("xcds".to_string(), Number::from(4));
        t.overlay(&over).unwrap();
        assert_eq!(t.get("xcds"), Some(&Number::from(4)));

        let mut bad = BTreeMap::new();
        bad.insert("xcd".to_string(), Number::from(4));
        let unknown = t.overlay(&bad).unwrap_err();
        assert_eq!(unknown, vec!["xcd".to_string()]);
        // And nothing was applied.
        assert!(!t.contains_key("xcd"));
    }

    /// A refused overlay must leave the table exactly as it was, or the
    /// error reports one machine and the run uses another.
    #[test]
    fn a_refused_overlay_applies_none_of_itself() {
        let mut t = table();
        let mut mixed = BTreeMap::new();
        mixed.insert("xcds".to_string(), Number::from(4));
        mixed.insert("nonsense".to_string(), Number::from(1));
        assert!(t.overlay(&mixed).is_err());
        assert_eq!(t.get("xcds"), Some(&Number::from(8)));
    }

    #[test]
    fn the_clock_is_lifted_out_of_the_machine_block() {
        let block = table().to_config_block().unwrap();
        assert_eq!(block["enabled"], Value::Bool(true));
        assert_eq!(block["clock_mhz"], serde_json::json!(2700));
        assert!(block["machine"].get(CLOCK_KEY).is_none());
        assert_eq!(block["machine"]["xcds"], serde_json::json!(8));
        // Rates stay rates and counts stay counts: an integer written as
        // a float is a config the emulator's parser may refuse.
        assert!(block["machine"]["dram.bytes_per_cycle"].is_f64());
        assert!(block["machine"]["dram.latency_cycles"].is_u64());
    }

    #[test]
    fn a_table_with_no_clock_cannot_be_emitted() {
        let t = TimingTable::from_pairs([("xcds".to_string(), Number::from(8))]);
        assert!(t.to_config_block().unwrap_err().contains(CLOCK_KEY));
    }

    #[test]
    fn options_round_trip_through_a_profile() {
        let mut options = SimpleMap::new();
        let mut over = BTreeMap::new();
        over.insert("xcds".to_string(), Number::from(4));
        over.insert(
            "dram.bytes_per_cycle".to_string(),
            Number::from_f64(1.5).unwrap(),
        );
        write_options(&mut options, true, &over);
        assert!(enabled_in_options(&options));
        assert_eq!(overrides_from_options(&options).unwrap(), over);

        clear_options(&mut options);
        assert!(!enabled_in_options(&options));
        assert!(overrides_from_options(&options).unwrap().is_empty());
    }

    /// A profile is hand-editable, so the natural JSON spelling has to
    /// work as well as the one mirage writes.
    #[test]
    fn a_hand_written_override_is_read_as_a_number() {
        let mut options = SimpleMap::new();
        options.insert(
            format!("{TIMING_OVERRIDE_PREFIX}xcds"),
            SimpleValue::Number(4),
        );
        assert_eq!(
            overrides_from_options(&options).unwrap()["xcds"],
            Number::from(4)
        );

        options.insert(
            format!("{TIMING_OVERRIDE_PREFIX}xcds"),
            SimpleValue::Boolean(true),
        );
        let e = overrides_from_options(&options).unwrap_err();
        assert!(e.contains("timing.machine.xcds"), "{e}");
    }

    #[test]
    fn a_disabled_write_leaves_nothing_behind() {
        let mut options = SimpleMap::new();
        let mut over = BTreeMap::new();
        over.insert("xcds".to_string(), Number::from(4));
        write_options(&mut options, true, &over);
        write_options(&mut options, false, &over);
        assert!(options.is_empty());
    }

    #[test]
    fn an_unknown_key_is_blamed_beside_its_neighbours() {
        let msg = unknown_keys_message(
            &["dram.latency_cyles".to_string()],
            &table(),
            "/tmp/tuning.json",
        );
        assert!(msg.contains("/tmp/tuning.json"), "{msg}");
        assert!(msg.contains("dram.latency_cycles"), "{msg}");
        assert!(msg.contains("dram.bytes_per_cycle"), "{msg}");
    }
}
