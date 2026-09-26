//! Linux host identity and cache facts shared by runtime frontends.
//!
//! The parser does not assign public API handles or names to cache records.
//! Frontends translate these facts into their own ABI representations.

use std::path::{Path, PathBuf};

/// CPU identity reported by Linux procfs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CpuInfo {
    /// First nonempty model name in `/proc/cpuinfo`.
    pub name: String,
    /// Number of processor entries in `/proc/cpuinfo`.
    pub compute_units: u32,
}

/// Cache type reported by Linux sysfs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CpuCacheKind {
    /// Data cache.
    Data,
    /// Instruction cache.
    Instruction,
    /// Unified cache.
    Unified,
    /// Another or unknown cache type.
    Other,
}

/// CPU cache reported by Linux sysfs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CpuCache {
    /// CPU directory that reported this cache.
    pub cpu: u32,
    /// First CPU in `shared_cpu_list`, if that list was valid.
    pub first_shared_cpu: Option<u32>,
    /// Cache type.
    pub kind: CpuCacheKind,
    /// Cache level.
    pub level: u32,
    /// Cache capacity in bytes.
    pub size_bytes: u32,
}

fn parse_memory_bytes(contents: &str) -> Option<usize> {
    let kilobytes = contents.lines().find_map(|line| {
        let mut fields = line.split_ascii_whitespace();
        (fields.next()? == "MemTotal:")
            .then(|| fields.next()?.parse::<u64>().ok())
            .flatten()
    })?;
    usize::try_from(kilobytes.checked_mul(1024)?).ok()
}

/// Returns Linux host memory capacity in bytes, if procfs reports it.
#[must_use]
pub fn memory_bytes() -> Option<usize> {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|contents| parse_memory_bytes(&contents))
        .filter(|bytes| *bytes != 0)
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

/// Returns Linux CPU identity, if procfs provides a model and processor count.
#[must_use]
pub fn cpu_info() -> Option<CpuInfo> {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| parse_cpu_info(&contents))
}

fn numeric_directories(root: &Path, prefix: &str) -> Vec<(u32, PathBuf)> {
    let mut entries = std::fs::read_dir(root)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let ordinal = name.strip_prefix(prefix)?.parse().ok()?;
            Some((ordinal, entry.path()))
        })
        .collect::<Vec<_>>();
    entries.sort_unstable_by_key(|(ordinal, _)| *ordinal);
    entries
}

fn first_cpu(contents: &str) -> Option<u32> {
    let contents = contents.trim_start();
    let length = contents.bytes().take_while(u8::is_ascii_digit).count();
    contents.get(..length)?.parse().ok()
}

fn parse_cache_size(contents: &str) -> Option<u32> {
    let contents = contents.trim();
    let length = contents.bytes().take_while(u8::is_ascii_digit).count();
    let value = contents.get(..length)?.parse::<u32>().ok()?;
    let multiplier = match contents.get(length..)?.trim() {
        "" => 1,
        "K" => 1024,
        "M" => 1024 * 1024,
        "G" => 1024 * 1024 * 1024,
        _ => return None,
    };
    value.checked_mul(multiplier)
}

fn discover_caches(root: &Path) -> Vec<CpuCache> {
    let mut caches = Vec::new();
    for (cpu, cpu_path) in numeric_directories(root, "cpu") {
        for (_, cache_path) in numeric_directories(&cpu_path.join("cache"), "index") {
            let first_shared_cpu = std::fs::read_to_string(cache_path.join("shared_cpu_list"))
                .ok()
                .and_then(|contents| first_cpu(&contents));
            let Some(kind) = std::fs::read_to_string(cache_path.join("type"))
                .ok()
                .map(|kind| match kind.trim() {
                    "Data" => CpuCacheKind::Data,
                    "Instruction" => CpuCacheKind::Instruction,
                    "Unified" => CpuCacheKind::Unified,
                    _ => CpuCacheKind::Other,
                })
            else {
                continue;
            };
            let Some(level) = std::fs::read_to_string(cache_path.join("level"))
                .ok()
                .and_then(|value| value.trim().parse::<u32>().ok())
            else {
                continue;
            };
            let Some(size_bytes) = std::fs::read_to_string(cache_path.join("size"))
                .ok()
                .and_then(|value| parse_cache_size(&value))
            else {
                continue;
            };
            caches.push(CpuCache {
                cpu,
                first_shared_cpu,
                kind,
                level,
                size_bytes,
            });
        }
    }
    caches
}

/// Returns Linux CPU cache records from all CPU directories in sysfs.
#[must_use]
pub fn caches() -> Vec<CpuCache> {
    discover_caches(Path::new("/sys/devices/system/cpu"))
}

/// Returns Linux CPU cache records for one NUMA node, if that node exists.
#[must_use]
pub fn caches_for_numa_node(node: u32) -> Option<Vec<CpuCache>> {
    let root = Path::new("/sys/devices/system/node").join(format!("node{node}"));
    root.is_dir().then(|| discover_caches(&root))
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
                "rocddi-host-cache-{}-{}",
                std::process::id(),
                NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn cache(&self, cpu: u32, index: u32, shared: &str, kind: &str, level: u32, size: &str) {
            let path = self.0.join(format!("cpu{cpu}/cache/index{index}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("shared_cpu_list"), shared).unwrap();
            std::fs::write(path.join("type"), kind).unwrap();
            std::fs::write(path.join("level"), level.to_string()).unwrap();
            std::fs::write(path.join("size"), size).unwrap();
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn cpu_info_requires_name_and_processor_count() {
        assert_eq!(
            parse_cpu_info("processor : 0\nmodel name : AMD Example CPU\nprocessor : 1\n"),
            Some(CpuInfo {
                name: "AMD Example CPU".to_owned(),
                compute_units: 2,
            })
        );
        assert_eq!(parse_cpu_info("processor : 0\n"), None);
        assert_eq!(parse_cpu_info("model name : AMD Example CPU\n"), None);
    }

    #[test]
    fn memory_capacity_is_reported_in_bytes() {
        assert_eq!(
            parse_memory_bytes("MemFree: 1024 kB\nMemTotal: 263219812 kB\nMemAvailable: 2048 kB\n"),
            Some(263_219_812 * 1024)
        );
        assert_eq!(parse_memory_bytes("MemFree: 1024 kB\n"), None);
        assert_eq!(parse_memory_bytes("MemTotal: invalid kB\n"), None);
    }

    #[test]
    fn cache_discovery_reports_all_types_and_sharing() {
        let fixture = Fixture::new();
        fixture.cache(0, 0, "0-1\n", "Data\n", 1, "48K\n");
        fixture.cache(0, 1, "0-1\n", "Instruction\n", 1, "32K\n");
        fixture.cache(0, 2, "0-3\n", "Unified\n", 2, "1M\n");
        fixture.cache(1, 0, "0-1\n", "Data\n", 1, "48K\n");
        assert_eq!(
            discover_caches(&fixture.0),
            vec![
                CpuCache {
                    cpu: 0,
                    first_shared_cpu: Some(0),
                    kind: CpuCacheKind::Data,
                    level: 1,
                    size_bytes: 48 * 1024,
                },
                CpuCache {
                    cpu: 0,
                    first_shared_cpu: Some(0),
                    kind: CpuCacheKind::Instruction,
                    level: 1,
                    size_bytes: 32 * 1024,
                },
                CpuCache {
                    cpu: 0,
                    first_shared_cpu: Some(0),
                    kind: CpuCacheKind::Unified,
                    level: 2,
                    size_bytes: 1024 * 1024,
                },
                CpuCache {
                    cpu: 1,
                    first_shared_cpu: Some(0),
                    kind: CpuCacheKind::Data,
                    level: 1,
                    size_bytes: 48 * 1024,
                },
            ]
        );
    }
}
