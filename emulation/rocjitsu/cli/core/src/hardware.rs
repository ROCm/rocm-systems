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
/// KFD topology exposes on this host, in [`kfd_gpus`] order. Returns an
/// empty vector when no AMD GPU is present or the KFD interface is
/// unavailable. CPU-only nodes are excluded.
///
/// A projection of [`kfd_gpus`] rather than a second walk of the same
/// sysfs tree. The two had drifted in three ways, and every one of them
/// was a way for two backends to disagree about the same machine: this
/// read only the class view, so a container that mounts just
/// `/sys/devices/virtual/kfd` had GPUs in DBT guest mode and none in
/// HotSwap or `rocjitsu-dbt`; it returned nodes in `readdir` order,
/// which is arbitrary; and it identified CPU nodes by a zero
/// `gfx_target_version` where the other uses a zero `gpu_id`.
#[must_use]
pub fn gpu_gfx_versions() -> Vec<u32> {
    kfd_gpus()
        .into_iter()
        .map(|gpu| gpu.gfx_target_version)
        // A GPU node that does not name its architecture tells a caller
        // asking only about architectures nothing.
        .filter(|version| *version != 0)
        .collect()
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
    match kfd_nodes_root(&[Path::new(KFD_NODES_DEVICE), Path::new(KFD_NODES)]) {
        Some(root) => kfd_gpus_from(root),
        None => Vec::new(),
    }
}

/// The first of `roots` that exists.
///
/// Split out so the fallback itself can be tested: which of the two
/// sysfs views is present is a property of the host — or of what a
/// container was given — and is not something a test can arrange for the
/// real paths.
fn kfd_nodes_root<'a>(roots: &[&'a Path]) -> Option<&'a Path> {
    roots.iter().copied().find(|root| root.is_dir())
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// Write a KFD node directory as sysfs presents one.
    fn node(root: &Path, id: u32, gpu_id: u32, gfx: u64, unique: u64) {
        let dir = root.join(id.to_string());
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("gpu_id"), format!("{gpu_id}\n")).unwrap();
        fs::write(
            dir.join("properties"),
            format!("simd_count 304\ngfx_target_version {gfx}\nunique_id {unique}\n"),
        )
        .unwrap();
    }

    #[test]
    fn parses_gfx_target_version_line() {
        let props = "cpu_cores_count 0\ngfx_target_version 90402\nsimd_count 304\n";
        assert_eq!(parse_property(props, "gfx_target_version"), Some(90402));
    }

    #[test]
    fn missing_gfx_target_version_is_none() {
        assert_eq!(
            parse_property("cpu_cores_count 16\n", "gfx_target_version"),
            None
        );
    }

    #[test]
    fn renders_gfx_names() {
        assert_eq!(gfx_name(90402), "gfx942");
        assert_eq!(gfx_name(90500), "gfx950");
        assert_eq!(gfx_name(90010), "gfx90a");
    }

    /// The architecture list is the GPU list, in the same order.
    ///
    /// These were two walks of the same sysfs tree and they disagreed:
    /// this one returned `readdir` order, so the architectures came back
    /// in an order that need not match the ordinals every visibility
    /// selector is written against.
    #[test]
    fn architectures_come_back_in_kfd_order() {
        let dir = std::env::temp_dir().join(format!("rocjitsu-hw-order-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        // Out of order on purpose, and with the CPU node in the middle.
        node(&dir, 4, 0x9500, 90500, 0x2222);
        node(&dir, 0, 0, 0, 0);
        node(&dir, 2, 0x9400, 90402, 0x1111);

        let gpus = kfd_gpus_from(&dir);
        let versions: Vec<u32> = gpus.iter().map(|g| g.gfx_target_version).collect();
        assert_eq!(
            versions,
            vec![90402, 90500],
            "node 2 before node 4, and the CPU node in neither"
        );

        fs::remove_dir_all(&dir).ok();
    }

    /// Both views of the same topology are consulted, device path first.
    ///
    /// A container may be given only one of them. Reading just the class
    /// view is what made such a container look GPU-less to HotSwap and
    /// `rocjitsu-dbt` while DBT guest mode, which asks the other
    /// function, saw its GPUs.
    #[test]
    fn the_device_view_is_tried_before_the_class_view() {
        let dir = std::env::temp_dir().join(format!("rocjitsu-hw-roots-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        let device = dir.join("device");
        let class = dir.join("class");
        let absent = dir.join("absent");
        fs::create_dir_all(&device).unwrap();
        fs::create_dir_all(&class).unwrap();

        assert_eq!(
            kfd_nodes_root(&[&device, &class]),
            Some(device.as_path()),
            "the device view wins when both are present"
        );
        assert_eq!(
            kfd_nodes_root(&[&absent, &class]),
            Some(class.as_path()),
            "the class view answers when the device view is absent"
        );
        assert_eq!(
            kfd_nodes_root(&[&absent]),
            None,
            "a host with neither has no KFD topology"
        );

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn missing_root_yields_empty() {
        let missing = std::env::temp_dir().join("rocjitsu-hw-does-not-exist-xyz");
        assert!(kfd_gpus_from(&missing).is_empty());
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
