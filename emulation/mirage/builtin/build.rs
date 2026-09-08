//! Embed the rocjitsu presets the builtin agents are, validating each
//! one against [`mirage_core::agent::AgentDef`] here rather than at run
//! time: the presets are source, not installed data — a released mirage
//! cannot go looking for them — and doing it here means a preset that
//! stops parsing fails the build instead of every run.

// A build script may panic: a preset that is missing, unparseable or no
// longer an agent is not something the crate can be built without, and
// there is no caller to return an error to.
#![allow(clippy::expect_used, clippy::panic)]

use std::fmt::Write as _;
use std::path::PathBuf;

/// The preset each builtin agent is, in `agents()` order: the Rust
/// constant to emit, and the `rocjitsu/configs/<stem>.json` it is read
/// from.
///
/// \NPI new GPU: add its preset here and a builtin agent in `agents.rs`.
const PRESETS: [(&str, &str); 3] = [
    ("MI300X", "gfx942_cdna3"),
    ("MI350X", "gfx950_mi355x"),
    ("MI450X", "gfx1250_mi455x"),
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let configs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../rocjitsu/configs");
    let mut output = String::new();
    for (name, stem) in PRESETS {
        let path = configs.join(format!("{stem}.json"));
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let config: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("invalid JSON in {}: {error}", path.display()));
        let agent = serde_json::json!({"vm": config["vm"], "topology": config["topology"]});
        serde_json::from_value::<mirage_core::agent::AgentDef>(agent.clone())
            .unwrap_or_else(|error| panic!("invalid agent in {}: {error}", path.display()));
        let json = serde_json::to_string(&agent).expect("agent JSON is serializable");
        writeln!(output, "pub(crate) const {name}: &str = {json:?};")
            .expect("writing to a String cannot fail");
    }
    let destination =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("presets.rs");
    std::fs::write(&destination, output)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", destination.display()));
}
