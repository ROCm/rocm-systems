// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux procfs and sysfs collection for passive host topology.
//!
//! This module assigns no runtime handles and does not activate a device.

use super::{
    CpuCache, CpuCacheKind, CpuInfo, CpuSet, HostCaches, HostDomain, HostTopologySnapshot,
};
use crate::topology::CacheInfo;
use std::path::{Path, PathBuf};

fn parse_memory_bytes(contents: &str) -> Option<usize> {
    let kilobytes = contents.lines().find_map(|line| {
        let mut fields = line.split_ascii_whitespace();
        (fields.next()? == "MemTotal:")
            .then(|| fields.next()?.parse::<u64>().ok())
            .flatten()
    })?;
    usize::try_from(kilobytes.checked_mul(1024)?).ok()
}

fn parse_cpu_info(contents: &str) -> Option<CpuInfo> {
    let mut name = None;
    let mut compute_units = 0_usize;
    for line in contents.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim() {
            "processor" if value.trim().parse::<u32>().is_ok() => {
                compute_units = compute_units.checked_add(1)?;
            }
            "model name" if name.is_none() && !value.trim().is_empty() => {
                name = Some(value.trim().to_owned());
            }
            _ => (),
        }
    }
    Some(CpuInfo {
        name: name?,
        compute_units: u32::try_from(compute_units)
            .ok()
            .filter(|count| *count != 0)?,
    })
}

fn parse_cpu_set(contents: &str) -> Option<CpuSet> {
    let contents = contents.trim();
    if contents.is_empty() {
        return None;
    }
    let mut ranges = Vec::new();
    for entry in contents.split(',') {
        let entry = entry.trim();
        let (first, last) = if let Some((first, last)) = entry.split_once('-') {
            (first.parse::<u32>().ok()?, last.parse::<u32>().ok()?)
        } else {
            let cpu = entry.parse::<u32>().ok()?;
            (cpu, cpu)
        };
        if first > last
            || ranges
                .last()
                .is_some_and(|&(_, previous)| first <= previous)
        {
            return None;
        }
        ranges.push((first, last));
    }
    Some(CpuSet { ranges })
}

fn parse_node_cpu_set(contents: &str) -> Option<CpuSet> {
    if contents.trim().is_empty() {
        Some(CpuSet { ranges: Vec::new() })
    } else {
        parse_cpu_set(contents)
    }
}

fn numeric_directories(root: &Path, prefix: &str) -> (Vec<(u32, PathBuf)>, bool) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return (Vec::new(), false);
    };
    let mut complete = true;
    let mut numbered = Vec::new();
    for entry in entries {
        let Ok(entry) = entry else {
            complete = false;
            continue;
        };
        let name = entry.file_name();
        let Some(ordinal) = name
            .to_str()
            .and_then(|name| name.strip_prefix(prefix))
            .and_then(|suffix| suffix.parse::<u32>().ok())
        else {
            continue;
        };
        numbered.push((ordinal, entry.path()));
    }
    numbered.sort_unstable_by_key(|(ordinal, _)| *ordinal);
    (numbered, complete)
}

fn parse_cache_size(contents: &str) -> Option<u64> {
    let contents = contents.trim();
    let length = contents.bytes().take_while(u8::is_ascii_digit).count();
    let value = contents.get(..length)?.parse::<u64>().ok()?;
    let multiplier = match contents.get(length..)?.trim() {
        "" => 1,
        "K" => 1024,
        "M" => 1024 * 1024,
        "G" => 1024 * 1024 * 1024,
        _ => return None,
    };
    value.checked_mul(multiplier)
}

fn discover_caches(root: &Path) -> HostCaches {
    let (cpus, mut complete) = numeric_directories(root, "cpu");
    let mut records = Vec::new();
    for (cpu, cpu_path) in cpus {
        let (indices, indices_complete) = numeric_directories(&cpu_path.join("cache"), "index");
        complete &= indices_complete;
        for (_, cache_path) in indices {
            let shared_cpus = std::fs::read_to_string(cache_path.join("shared_cpu_list"))
                .ok()
                .and_then(|contents| parse_cpu_set(&contents));
            let line_size_bytes = std::fs::read_to_string(cache_path.join("coherency_line_size"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
                .filter(|size| *size != 0);
            complete &= shared_cpus.is_some() && line_size_bytes.is_some();
            let Some(kind) = std::fs::read_to_string(cache_path.join("type"))
                .ok()
                .map(|kind| match kind.trim() {
                    "Data" => CpuCacheKind::Data,
                    "Instruction" => CpuCacheKind::Instruction,
                    "Unified" => CpuCacheKind::Unified,
                    _ => CpuCacheKind::Other,
                })
            else {
                complete = false;
                continue;
            };
            let Some(level) = std::fs::read_to_string(cache_path.join("level"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                complete = false;
                continue;
            };
            let Some(size_bytes) = std::fs::read_to_string(cache_path.join("size"))
                .ok()
                .and_then(|value| parse_cache_size(&value))
            else {
                complete = false;
                continue;
            };
            records.push(CpuCache {
                cpu,
                shared_cpus,
                info: CacheInfo::new(level, size_bytes, line_size_bytes, kind),
            });
        }
    }
    HostCaches { records, complete }
}

fn snapshot_from(
    meminfo: &Path,
    cpuinfo: &Path,
    online: &Path,
    cpu_root: &Path,
    node_root: &Path,
) -> HostTopologySnapshot {
    let memory_bytes = std::fs::read_to_string(meminfo)
        .ok()
        .and_then(|contents| parse_memory_bytes(&contents))
        .filter(|bytes| *bytes != 0);
    let cpu = std::fs::read_to_string(cpuinfo)
        .ok()
        .and_then(|contents| parse_cpu_info(&contents));
    let online_cpus = std::fs::read_to_string(online)
        .ok()
        .and_then(|contents| parse_cpu_set(&contents));
    let caches = discover_caches(cpu_root);
    let (entries, mut domains_complete) = if node_root.exists() {
        numeric_directories(node_root, "node")
    } else {
        (Vec::new(), true)
    };
    let mut domains = Vec::with_capacity(entries.len());
    for (numa_node, path) in entries {
        let cpus = std::fs::read_to_string(path.join("cpulist"))
            .ok()
            .and_then(|contents| parse_node_cpu_set(&contents));
        // The node CPU entries normally link to the same cache directories as
        // cpu_root. Reuse a complete global scan instead of reading every
        // cache file once more for each NUMA node. A partial scan or missing
        // CPU list still gets the node-local best-effort scan.
        let node_caches = if let Some(cpus) = cpus.as_ref().filter(|_| caches.complete) {
            HostCaches {
                records: caches
                    .records
                    .iter()
                    .filter(|cache| cpus.contains(cache.cpu))
                    .cloned()
                    .collect(),
                complete: true,
            }
        } else {
            discover_caches(&path)
        };
        domains_complete &= cpus.is_some() && node_caches.complete;
        domains.push(HostDomain {
            numa_node,
            cpus,
            caches: node_caches,
        });
    }
    HostTopologySnapshot {
        memory_bytes,
        cpu,
        online_cpus,
        caches,
        domains,
        domains_complete,
    }
}

/// Reads Linux host facts without requiring KFD or an activated device.
#[must_use]
pub(super) fn snapshot() -> HostTopologySnapshot {
    snapshot_from(
        Path::new("/proc/meminfo"),
        Path::new("/proc/cpuinfo"),
        Path::new("/sys/devices/system/cpu/online"),
        Path::new("/sys/devices/system/cpu"),
        Path::new("/sys/devices/system/node"),
    )
}

pub(crate) fn attached_numa_node_for_render(
    render_root: &Path,
    node_root: &Path,
    online: &Path,
    render_minor: u32,
) -> Option<u32> {
    let node = std::fs::read_to_string(
        render_root.join(format!("renderD{render_minor}/device/numa_node")),
    )
    .ok()?
    .trim()
    .parse::<u32>()
    .ok()?;
    let cpus = std::fs::read_to_string(node_root.join(format!("node{node}/cpulist")))
        .ok()
        .and_then(|value| parse_node_cpu_set(&value))?;
    let online = std::fs::read_to_string(online)
        .ok()
        .and_then(|value| parse_cpu_set(&value))?;
    cpus.intersects(&online).then_some(node)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    static NEXT_FIXTURE: AtomicU32 = AtomicU32::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "rocddi-host-topology-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn write(&self, path: &str, value: &str) {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, value).unwrap();
        }

        fn cache(&self, root: &str, cpu: u32, index: u32, shared: &str, kind: &str, level: u32) {
            let path = format!("{root}/cpu{cpu}/cache/index{index}");
            self.write(&format!("{path}/shared_cpu_list"), shared);
            self.write(&format!("{path}/coherency_line_size"), "64\n");
            self.write(&format!("{path}/type"), kind);
            self.write(&format!("{path}/level"), &level.to_string());
            self.write(&format!("{path}/size"), "48K\n");
        }

        fn snapshot(&self) -> HostTopologySnapshot {
            snapshot_from(
                &self.0.join("meminfo"),
                &self.0.join("cpuinfo"),
                &self.0.join("online"),
                &self.0.join("cpu"),
                &self.0.join("node"),
            )
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cpu_sets_reject_malformed_ranges_without_expanding_them() {
        let set = parse_cpu_set("0-3,8,10-12\n").unwrap();
        assert_eq!(set.first(), Some(0));
        assert!(set.contains(11));
        assert!(!set.contains(9));
        assert!(set.intersects(&parse_cpu_set("9-10").unwrap()));
        assert!(!set.intersects(&parse_cpu_set("4-7").unwrap()));
        for malformed in ["", "2-1", "0-3,3-4", "3,2", "1-", "0-4294967296"] {
            assert_eq!(parse_cpu_set(malformed), None);
        }
    }

    #[test]
    fn cpu_info_and_memory_capacity_keep_unavailable_fields_distinct() {
        assert_eq!(
            parse_cpu_info("processor : 0\nmodel name : AMD Example CPU\nprocessor : 1\n"),
            Some(CpuInfo {
                name: "AMD Example CPU".to_owned(),
                compute_units: 2,
            })
        );
        assert_eq!(parse_cpu_info("processor : 0\n"), None);
        assert_eq!(parse_cpu_info("model name : AMD Example CPU\n"), None);
        assert_eq!(
            parse_memory_bytes("MemTotal: 263219812 kB\n"),
            Some(263_219_812 * 1024)
        );
        assert_eq!(parse_memory_bytes("MemTotal: invalid kB\n"), None);
    }

    #[test]
    fn snapshot_partitions_complete_cache_scan_by_numa_domain() {
        let fixture = Fixture::new();
        fixture.write("meminfo", "MemTotal: 1024 kB\n");
        fixture.write(
            "cpuinfo",
            "processor : 0\nmodel name : CPU\nprocessor : 1\n",
        );
        fixture.write("online", "0-1,4-5\n");
        fixture.cache("cpu", 0, 0, "0-1\n", "Data\n", 1);
        fixture.cache("cpu", 0, 1, "0-1\n", "Unified\n", 2);
        fixture.cache("cpu", 4, 0, "4-5\n", "Data\n", 1);
        fixture.write("node/node0/cpulist", "0-1\n");
        fixture.write("node/node2/cpulist", "4-5\n");
        let snapshot = fixture.snapshot();
        assert_eq!(snapshot.memory_bytes, Some(1024 * 1024));
        assert_eq!(snapshot.cpu.as_ref().unwrap().compute_units, 2);
        assert_eq!(snapshot.domains.len(), 2);
        assert_eq!(snapshot.domain(0).unwrap().caches.records.len(), 2);
        assert_eq!(snapshot.domain(0).unwrap().caches.records[0].cpu, 0);
        assert_eq!(snapshot.domain(2).unwrap().caches.records.len(), 1);
        assert_eq!(snapshot.domain(2).unwrap().caches.records[0].cpu, 4);
        assert!(
            snapshot
                .domain(2)
                .unwrap()
                .cpus
                .as_ref()
                .unwrap()
                .contains(5)
        );
        assert!(snapshot.domains_complete);
        assert_eq!(snapshot.caches.records.len(), 3);
        assert_eq!(snapshot.caches.records[0].first_shared_cpu(), Some(0));
        assert_eq!(snapshot.caches.records[0].info.line_size_bytes, Some(64));
        assert_eq!(snapshot.caches.records[1].info.kind, CpuCacheKind::Unified);
        assert!(snapshot.caches.complete);
    }

    #[test]
    fn memory_only_numa_node_has_complete_empty_cpu_and_cache_sets() {
        let fixture = Fixture::new();
        fixture.write("online", "0\n");
        fixture.cache("cpu", 0, 0, "0\n", "Data\n", 1);
        fixture.write("node/node0/cpulist", "0\n");
        fixture.write("node/node2/cpulist", "\n");

        let snapshot = fixture.snapshot();
        assert!(snapshot.domains_complete);
        let node = snapshot.domain(2).unwrap();
        assert_eq!(node.cpus.as_ref().unwrap().first(), None);
        assert_eq!(node.cpus.as_ref().unwrap().ranges, Vec::new());
        assert!(node.caches.complete);
        assert!(node.caches.records.is_empty());

        fixture.write("node/node2/cpulist", "invalid\n");
        let snapshot = fixture.snapshot();
        assert!(!snapshot.domains_complete);
        assert_eq!(snapshot.domain(2).unwrap().cpus, None);

        std::fs::remove_file(fixture.0.join("node/node2/cpulist")).unwrap();
        let snapshot = fixture.snapshot();
        assert!(!snapshot.domains_complete);
        assert_eq!(snapshot.domain(2).unwrap().cpus, None);
    }

    #[test]
    fn incomplete_cache_entry_does_not_hide_complete_records() {
        let fixture = Fixture::new();
        fixture.cache("cpu", 0, 0, "0-1\n", "Data\n", 1);
        fixture.cache("cpu", 0, 1, "0-1\n", "Data\n", 2);
        std::fs::remove_file(fixture.0.join("cpu/cpu0/cache/index1/size")).unwrap();
        let caches = discover_caches(&fixture.0.join("cpu"));
        assert_eq!(caches.records.len(), 1);
        assert!(!caches.complete);
    }

    #[test]
    fn incomplete_global_scan_uses_node_local_cache_records() {
        let fixture = Fixture::new();
        fixture.cache("cpu", 0, 0, "0\n", "Data\n", 1);
        std::fs::remove_file(fixture.0.join("cpu/cpu0/cache/index0/size")).unwrap();
        fixture.write("node/node0/cpulist", "0\n");
        fixture.cache("node/node0", 0, 0, "0\n", "Data\n", 1);

        let snapshot = fixture.snapshot();
        assert!(!snapshot.caches.complete);
        assert!(snapshot.domain(0).unwrap().caches.complete);
        assert_eq!(snapshot.domain(0).unwrap().caches.records.len(), 1);
        assert!(snapshot.domains_complete);
    }

    #[test]
    fn endpoint_locality_requires_an_online_cpu_on_the_os_numa_node() {
        let fixture = Fixture::new();
        fixture.write("render/renderD128/device/numa_node", "2\n");
        fixture.write("node/node2/cpulist", "4-7\n");
        fixture.write("online", "0-3\n");
        let query = || {
            attached_numa_node_for_render(
                &fixture.0.join("render"),
                &fixture.0.join("node"),
                &fixture.0.join("online"),
                128,
            )
        };
        assert_eq!(query(), None);
        fixture.write("online", "0-4\n");
        assert_eq!(query(), Some(2));
        fixture.write("render/renderD128/device/numa_node", "-1\n");
        assert_eq!(query(), None);
    }
}
