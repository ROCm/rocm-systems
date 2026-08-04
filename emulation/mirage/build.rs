//! Build script: generate the third-party dependency/license manifest
//! that `mirage about` prints.
//!
//! At build time we ask cargo for the fully-resolved dependency graph
//! (`cargo metadata`) and distil it into a small, sorted text manifest
//! of every crate mirage links, with its version and SPDX license. The
//! manifest is written to `OUT_DIR/about.txt` and embedded into the
//! binary via `include_str!`, so `mirage about` needs no files at
//! runtime and the list can never drift from what was actually built.
//!
//! The generation is best-effort: if `cargo metadata` is unavailable
//! (e.g. a restricted offline build) we still emit a valid, if sparse,
//! manifest so the build never fails on account of `about`.

use std::path::PathBuf;

fn main() {
    // Only regenerate when the resolved dependency set changes.
    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let manifest = generate_manifest().unwrap_or_else(|e| {
        println!("cargo:warning=mirage about: could not build dependency manifest: {e}");
        String::from("(third-party dependency manifest unavailable for this build)\n")
    });
    std::fs::write(out_dir.join("about.txt"), manifest).expect("write about.txt");
}

/// Run `cargo metadata` and render the dependency/license manifest.
fn generate_manifest() -> Result<String, String> {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = std::process::Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--all-features"])
        .output()
        .map_err(|e| format!("spawn cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo metadata exited with {}", output.status));
    }

    let value: serde_json::Value =
        serde_json::from_slice(&output.stdout).map_err(|e| format!("parse metadata json: {e}"))?;
    let packages = value
        .get("packages")
        .and_then(|p| p.as_array())
        .ok_or("metadata has no packages array")?;

    // Collect (name, version, license) for every resolved crate, skipping
    // the mirage workspace crates themselves (they're first-party).
    let mut entries: Vec<(String, String, String)> = Vec::new();
    for pkg in packages {
        let name = pkg.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if name.is_empty()
            || name == "mirage"
            || name.starts_with("mirage_")
            || name == "rocjitsu_sys"
        {
            continue;
        }
        let version = pkg.get("version").and_then(|v| v.as_str()).unwrap_or("");
        let license = pkg
            .get("license")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| {
                pkg.get("license_file")
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                    .map(|f| format!("see {f}"))
            })
            .unwrap_or_else(|| "(unspecified)".to_string());
        entries.push((name.to_string(), version.to_string(), license));
    }

    entries.sort();
    entries.dedup();

    let mut out = String::new();
    out.push_str(&format!(
        "mirage links {} third-party crate(s):\n\n",
        entries.len()
    ));
    for (name, version, license) in &entries {
        out.push_str(&format!("  {name} {version} — {license}\n"));
    }
    Ok(out)
}
