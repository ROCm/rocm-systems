//! Keep host scheduling policy in the rocjitsu backend, sourced from its presets.
#![allow(clippy::expect_used, clippy::panic)]

use std::fmt::Write as _;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let configs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../rocjitsu/configs");
    let mut out = String::from(
        "// Generated from rocjitsu configs by build.rs.\n\
         fn target_thread_allocations(gfx_target_version: u32) -> Vec<ExecutionThreadChoice> {\n\
         match gfx_target_version {\n",
    );
    for stem in [
        "gfx90a_mi210_kmd",
        "gfx942_cdna3",
        "gfx950_mi355x",
        "gfx1100_w7900",
        "gfx1151",
        "gfx1201_r9700",
        "gfx1250_mi455x",
    ] {
        let path = configs.join(format!("{stem}.json"));
        println!("cargo:rerun-if-changed={}", path.display());
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let json: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("invalid JSON in {}: {e}", path.display()));
        let target = json["vm"]["gpu"]["device"]["gfx_target_version"]
            .as_u64()
            .expect("preset must have a numeric gfx_target_version");
        writeln!(out, "{target} => vec![").expect("writing to a String cannot fail");
        for choice in json["thread_allocations"]
            .as_array()
            .expect("preset must have thread_allocations")
        {
            let engines = choice["num_threads"]
                .as_u64()
                .expect("granule must have num_threads");
            let dispatch = choice["cpu_dispatch_threads"]
                .as_u64()
                .expect("granule must have cpu_dispatch_threads");
            assert!(
                engines > 0 && dispatch > 0,
                "thread widths must be positive"
            );
            writeln!(out, "ExecutionThreadChoice {{ num_threads: {engines}, cpu_dispatch_threads: {dispatch} }},")
                .expect("writing to a String cannot fail");
        }
        out.push_str("],\n");
    }
    out.push_str("_ => Vec::new(),\n}\n}\n");
    let dest = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"))
        .join("thread_allocations.rs");
    std::fs::write(&dest, out).unwrap_or_else(|e| panic!("cannot write {}: {e}", dest.display()));
}
