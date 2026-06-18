//! Generic host GPU detection.
//!
//! Emulator backends sometimes require specific physical hardware to
//! be present (for example, HotSwap can only retarget code onto a real
//! GPU of a compatible architecture). This module exposes a small,
//! emulator-agnostic way to enumerate the AMD GPUs the kernel reports,
//! so each backend can decide for itself whether the host is
//! supported.
//!
//! Detection uses the kernel's KFD (Kernel Fusion Driver) sysfs
//! topology, which lists every compute node along with its
//! `gfx_target_version` — a packed decimal encoding of the GPU's gfx
//! architecture (e.g. `90402` for `gfx942`, `90500` for `gfx950`).
//! CPU-only nodes report `gfx_target_version 0` and are skipped.
//!
//! On hosts without an AMD GPU or KFD interface (CI, non-Linux, no
//! driver) the enumeration simply returns an empty list rather than
//! failing.

use std::fs;
use std::path::Path;

/// The KFD topology nodes directory in sysfs.
const KFD_NODES: &str = "/sys/class/kfd/kfd/topology/nodes";

/// Enumerate the `gfx_target_version` of every GPU node the kernel's
/// KFD topology exposes on this host. Returns an empty vector when no
/// AMD GPU is present or the KFD interface is unavailable. CPU-only
/// nodes (`gfx_target_version 0`) are excluded.
pub fn gpu_gfx_versions() -> Vec<u32> {
    detect_from(Path::new(KFD_NODES))
}

/// Render a packed `gfx_target_version` as a conventional `gfxNNN`
/// architecture string. The encoding is decimal `MMMmmpp` (major,
/// minor, step), so `90402` → `gfx942` and `90500` → `gfx950`. Step
/// values above 9 are rendered in hex (the gfx convention), e.g.
/// `gfx90a`.
pub fn gfx_name(gfx_target_version: u32) -> String {
    let major = gfx_target_version / 10000;
    let minor = (gfx_target_version / 100) % 100;
    let step = gfx_target_version % 100;
    format!("gfx{major}{minor}{step:x}")
}

/// Read the GPU nodes under `root`, collecting the non-zero
/// `gfx_target_version` of each. Split out from [`gpu_gfx_versions`] so
/// it can be exercised against a fixture directory in tests.
fn detect_from(root: &Path) -> Vec<u32> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return out;
    };
    for entry in entries.flatten() {
        let props = entry.path().join("properties");
        let Ok(text) = fs::read_to_string(&props) else {
            continue;
        };
        if let Some(v) = parse_gfx_target_version(&text)
            && v != 0
        {
            out.push(v);
        }
    }
    out
}

/// Pull the `gfx_target_version` value out of a KFD node `properties`
/// file. Each line is `key value`; returns the value for the
/// `gfx_target_version` key when present and parseable.
fn parse_gfx_target_version(properties: &str) -> Option<u32> {
    for line in properties.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("gfx_target_version") {
            return parts.next().and_then(|v| v.parse::<u32>().ok());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gfx_target_version_line() {
        let props = "cpu_cores_count 0\ngfx_target_version 90402\nsimd_count 304\n";
        assert_eq!(parse_gfx_target_version(props), Some(90402));
    }

    #[test]
    fn missing_gfx_target_version_is_none() {
        assert_eq!(parse_gfx_target_version("cpu_cores_count 16\n"), None);
    }

    #[test]
    fn renders_gfx_names() {
        assert_eq!(gfx_name(90402), "gfx942");
        assert_eq!(gfx_name(90500), "gfx950");
        assert_eq!(gfx_name(90010), "gfx90a");
    }

    #[test]
    fn detect_skips_cpu_nodes_and_collects_gpus() {
        let dir = std::env::temp_dir().join(format!("mirage-hw-test-{}", std::process::id()));
        let node0 = dir.join("0");
        let node1 = dir.join("1");
        fs::create_dir_all(&node0).unwrap();
        fs::create_dir_all(&node1).unwrap();
        // CPU node: gfx_target_version 0 -> skipped.
        fs::write(
            node0.join("properties"),
            "cpu_cores_count 16\ngfx_target_version 0\n",
        )
        .unwrap();
        // GPU node.
        fs::write(
            node1.join("properties"),
            "simd_count 304\ngfx_target_version 90402\n",
        )
        .unwrap();

        let mut found = detect_from(&dir);
        found.sort_unstable();
        assert_eq!(found, vec![90402]);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_root_yields_empty() {
        let missing = std::env::temp_dir().join("mirage-hw-does-not-exist-xyz");
        assert!(detect_from(&missing).is_empty());
    }
}
