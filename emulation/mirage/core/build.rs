//! Embeds every `agents/*.json` from the mirage workspace into the
//! binary at compile time. Produces `$OUT_DIR/agent_builtins.rs`:
//!
//! ```text
//! pub const BUILTIN_AGENT_JSON: &[(&str, &str)] = &[
//!     ("MI300X", include_str!("/abs/path/MI300X.json")),
//!     ("MI350X", include_str!("/abs/path/MI350X.json")),
//! ];
//! ```
//!
//! Runtime code parses each blob as an `AgentDef`, then reserializes
//! so the on-disk file contains only fields mirage understands.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let agents_dir = manifest.join("..").join("agents");

    println!("cargo:rerun-if-changed={}", agents_dir.display());

    let mut entries: Vec<(String, PathBuf)> = Vec::new();
    if let Ok(read) = std::fs::read_dir(&agents_dir) {
        for entry in read.flatten() {
            let path = entry.path();
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            println!("cargo:rerun-if-changed={}", path.display());
            entries.push((stem.to_string(), path));
        }
    }
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let dst = out_dir.join("agent_builtins.rs");

    let mut src = String::new();
    src.push_str("pub const BUILTIN_AGENT_JSON: &[(&str, &str)] = &[\n");
    for (name, path) in &entries {
        src.push_str(&format!(
            "    ({:?}, include_str!({:?})),\n",
            name,
            path.display().to_string()
        ));
    }
    src.push_str("];\n");

    std::fs::write(&dst, src).expect("write agent_builtins.rs");
}
