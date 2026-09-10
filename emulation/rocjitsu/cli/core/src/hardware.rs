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

use crate::visibility::{self, VisibleGpu};

/// The KFD topology nodes directory in sysfs.
const KFD_NODES: &str = "/sys/class/kfd/kfd/topology/nodes";

/// Where KFD topology appears when the driver is bound directly. Tried
/// before [`KFD_NODES`], which is the symlinked class view of the same
/// thing; either answers, but a container may expose only one.
const KFD_NODES_DEVICE: &str = "/sys/devices/virtual/kfd/kfd/topology/nodes";

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

/// Parse a conventional `gfxNNN` architecture string back into its
/// packed `gfx_target_version`. The inverse of [`gfx_name`]: the last
/// two characters are hex nibbles (minor, step) and everything before
/// them is the decimal major, so `gfx942` → `90402` and `gfx90a` →
/// `90010`.
#[must_use]
pub fn gfx_target_version_from_name(name: &str) -> Option<u32> {
    let digits = name.strip_prefix("gfx")?;
    if digits.len() < 3 {
        return None;
    }
    let (major_text, tail) = digits.split_at(digits.len() - 2);
    let major: u32 = major_text.parse().ok()?;
    let mut nibbles = tail.chars().map(|c| c.to_digit(16));
    let minor = nibbles.next().flatten()?;
    let step = nibbles.next().flatten()?;
    Some(major * 10000 + minor * 100 + step)
}

/// Enumerate every GPU the kernel's KFD topology exposes, with the
/// identity ROCm visibility selectors are written against.
///
/// Ordered by node id and renumbered from zero with CPU nodes dropped,
/// which is what makes an ordinal here mean the same thing it means
/// inside ROCR. Returns an empty vector when no AMD GPU is present.
#[must_use]
pub fn kfd_gpus() -> Vec<VisibleGpu> {
    for root in [KFD_NODES_DEVICE, KFD_NODES] {
        let root = Path::new(root);
        if root.is_dir() {
            return kfd_gpus_from(root);
        }
    }
    Vec::new()
}

/// Read the GPU nodes under `root`. Split out from [`kfd_gpus`] so it
/// can be exercised against a fixture directory in tests.
fn kfd_gpus_from(root: &Path) -> Vec<VisibleGpu> {
    let mut nodes: Vec<(u32, VisibleGpu)> = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let Some(node_id) = path
            .file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| name.parse::<u32>().ok())
        else {
            continue;
        };
        // A zero `gpu_id` is a CPU node. Skipping it here rather than
        // leaving it to `enumerate_kfd_gpus` costs nothing and keeps the
        // properties read off the hot path for a machine with many CPU
        // nodes.
        let gpu_id = read_u32_file(&path.join("gpu_id")).unwrap_or(0);
        if gpu_id == 0 {
            continue;
        }
        let properties = fs::read_to_string(path.join("properties")).unwrap_or_default();
        nodes.push((
            node_id,
            VisibleGpu {
                ordinal: 0,
                gpu_id,
                gfx_target_version: parse_property(&properties, "gfx_target_version")
                    .and_then(|v| u32::try_from(v).ok())
                    .unwrap_or(0),
                unique_id: parse_property(&properties, "unique_id").unwrap_or(0),
            },
        ));
    }
    nodes.sort_unstable_by_key(|(node_id, _)| *node_id);
    let ordered: Vec<VisibleGpu> = nodes.into_iter().map(|(_, gpu)| gpu).collect();
    visibility::enumerate_kfd_gpus(&ordered)
}

/// Read a sysfs file holding a single unsigned integer.
fn read_u32_file(path: &Path) -> Option<u32> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Pull a numeric value out of a KFD node `properties` file, whose lines
/// are `key value` pairs.
fn parse_property(properties: &str, key: &str) -> Option<u64> {
    for line in properties.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some(key) {
            return parts.next().and_then(|v| v.parse::<u64>().ok());
        }
    }
    None
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
/// file.
fn parse_gfx_target_version(properties: &str) -> Option<u32> {
    parse_property(properties, "gfx_target_version").and_then(|v| u32::try_from(v).ok())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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
        let dir = std::env::temp_dir().join(format!("rocjitsu-hw-test-{}", std::process::id()));
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
        let missing = std::env::temp_dir().join("rocjitsu-hw-does-not-exist-xyz");
        assert!(detect_from(&missing).is_empty());
    }

    #[test]
    fn parses_gfx_names_back_to_versions() {
        assert_eq!(gfx_target_version_from_name("gfx942"), Some(90402));
        assert_eq!(gfx_target_version_from_name("gfx950"), Some(90500));
        assert_eq!(gfx_target_version_from_name("gfx90a"), Some(90010));
        assert_eq!(gfx_target_version_from_name("gfx1201"), Some(120001));
        assert_eq!(gfx_target_version_from_name("gfx"), None);
        assert_eq!(gfx_target_version_from_name("gfx9"), None);
        assert_eq!(gfx_target_version_from_name("sm_90"), None);
        assert_eq!(gfx_target_version_from_name("gfx9zz"), None);
    }

    #[test]
    fn gfx_name_round_trips() {
        for version in [90402, 90500, 90010, 120001, 110000] {
            assert_eq!(
                gfx_target_version_from_name(&gfx_name(version)),
                Some(version)
            );
        }
    }

    #[test]
    fn kfd_enumeration_orders_by_node_id_and_skips_cpu_nodes() {
        let dir = std::env::temp_dir().join(format!("rocjitsu-kfd-test-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();

        // Written out of order on purpose: readdir order is arbitrary,
        // and an ordinal that depends on it would silently disagree with
        // the one ROCR assigns.
        for (node, gpu_id, gfx, unique) in [
            (0u32, 0u32, 0u64, 0u64),
            (4, 0x9500, 90500, 0x2222_2222_2222_2222),
            (2, 0x9400, 90402, 0x1111_1111_1111_1111),
        ] {
            let node_dir = dir.join(node.to_string());
            fs::create_dir_all(&node_dir).unwrap();
            fs::write(node_dir.join("gpu_id"), format!("{gpu_id}\n")).unwrap();
            fs::write(
                node_dir.join("properties"),
                format!("simd_count 304\ngfx_target_version {gfx}\nunique_id {unique}\n"),
            )
            .unwrap();
        }

        let gpus = kfd_gpus_from(&dir);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].ordinal, 0);
        assert_eq!(gpus[0].gpu_id, 0x9400);
        assert_eq!(gpus[0].gfx_target_version, 90402);
        assert_eq!(gpus[0].unique_id, 0x1111_1111_1111_1111);
        assert_eq!(gpus[1].ordinal, 1);
        assert_eq!(gpus[1].gpu_id, 0x9500);

        fs::remove_dir_all(&dir).ok();
    }
}
