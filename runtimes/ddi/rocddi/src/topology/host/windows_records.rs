// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Checked decoding of the 64-bit Windows logical-processor record stream.
//! Records have individual byte lengths and variable affinity arrays; the
//! native union's Rust size is never used as a record stride.

use super::{CpuCache, CpuId, CpuInfo, CpuSet, HostCaches, HostDomain, HostTopologyGraphSnapshot};
use crate::topology::{CacheAssociations, CacheInfo, CacheKind};

fn word<const N: usize>(bytes: &[u8], offset: usize) -> Option<[u8; N]> {
    bytes.get(offset..offset.checked_add(N)?)?.try_into().ok()
}

fn u16_at(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(word(bytes, offset)?))
}

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(word(bytes, offset)?))
}

/// Affinities follow a 24-byte processor/NUMA or 32-byte cache prefix.
/// Legacy NUMA/cache records use one mask and leave `GroupCount` zero.
fn affinities(bytes: &[u8], offset: usize, count: usize) -> Option<CpuSet> {
    let masks = bytes.get(offset..offset.checked_add(count.checked_mul(16)?)?)?;
    let mut ranges = Vec::new();
    for mask in masks.chunks_exact(16) {
        let mut bits = u64::from_le_bytes(word(mask, 0)?);
        let group = u16_at(mask, 8)?;
        while bits != 0 {
            let first = bits.trailing_zeros();
            let run = (bits >> first).trailing_ones();
            let base = CpuId::windows(group, u8::try_from(first).ok()?)?.0;
            ranges.push((base, base + u64::from(run) - 1));
            bits &= if first + run == 64 {
                0
            } else {
                u64::MAX << (first + run)
            };
        }
    }
    let mut set = CpuSet { ranges };
    normalize(&mut set);
    Some(set)
}

fn normalize(set: &mut CpuSet) {
    set.ranges.sort_unstable();
    let mut used = 0;
    for index in 0..set.ranges.len() {
        let (first, last) = set.ranges[index];
        if used != 0 && first <= set.ranges[used - 1].1 + 1 {
            set.ranges[used - 1].1 = set.ranges[used - 1].1.max(last);
        } else {
            set.ranges[used] = (first, last);
            used += 1;
        }
    }
    set.ranges.truncate(used);
}

fn cache(bytes: &[u8]) -> Option<CpuCache> {
    let cpus = affinities(bytes, 32, usize::from(u16_at(bytes, 30)?).max(1))?;
    let level = u32::from(*bytes.first()?);
    let line = u32::from(u16_at(bytes, 2)?);
    let size = u64::from(u32_at(bytes, 4)?);
    if level == 0 || line == 0 || size == 0 {
        return None;
    }
    Some(CpuCache {
        cpu: cpus.first()?,
        shared_cpus: Some(cpus),
        info: CacheInfo {
            level,
            size_bytes: size,
            line_size_bytes: Some(line),
            kind: match u32_at(bytes, 8)? {
                0 => CacheKind::Unified,
                1 => CacheKind::Instruction,
                2 => CacheKind::Data,
                _ => CacheKind::Other,
            },
            associations: CacheAssociations::CPU,
        },
    })
}

pub(super) fn decode(bytes: &[u8]) -> HostTopologyGraphSnapshot {
    let mut result = HostTopologyGraphSnapshot {
        memory_bytes: None,
        cpu: None,
        cpu_apic_ids: Vec::new(),
        online_cpus: None,
        caches: HostCaches {
            records: Vec::new(),
            complete: true,
        },
        domains: Vec::new(),
        domains_complete: true,
    };
    let mut remaining = bytes;
    let mut cpus = CpuSet { ranges: Vec::new() };
    let mut processors_complete = true;
    while !remaining.is_empty() {
        let record = u32_at(remaining, 4)
            .and_then(|size| usize::try_from(size).ok())
            .filter(|size| *size >= 8)
            .and_then(|size| remaining.get(..size));
        let Some(record) = record else {
            processors_complete = false;
            result.caches.complete = false;
            result.domains_complete = false;
            break;
        };
        let body = &record[8..];
        match u32_at(record, 0) {
            Some(0) => {
                let members = u16_at(body, 22)
                    .filter(|count| *count != 0)
                    .and_then(|count| affinities(body, 24, usize::from(count)))
                    .filter(|set| set.first().is_some());
                if let Some(members) = members {
                    cpus.ranges.extend(members.ranges);
                } else {
                    processors_complete = false;
                }
            }
            Some(1 | 6) => {
                let domain = u32_at(body, 0).zip(
                    u16_at(body, 22)
                        .and_then(|count| affinities(body, 24, usize::from(count).max(1))),
                );
                if let Some((numa_node, cpus)) = domain
                    .filter(|(id, _)| !result.domains.iter().any(|domain| domain.numa_node == *id))
                {
                    result.domains.push(HostDomain {
                        numa_node,
                        cpus: Some(cpus),
                        caches: HostCaches {
                            records: Vec::new(),
                            complete: true,
                        },
                    });
                } else {
                    result.domains_complete = false;
                }
            }
            Some(2) => {
                if let Some(cache) = cache(body) {
                    result.caches.records.push(cache);
                } else {
                    result.caches.complete = false;
                }
            }
            _ => (),
        }
        remaining = &remaining[record.len()..];
    }
    finish(result, cpus, processors_complete)
}

fn finish(
    mut result: HostTopologyGraphSnapshot,
    mut cpus: CpuSet,
    processors_complete: bool,
) -> HostTopologyGraphSnapshot {
    normalize(&mut cpus);
    if processors_complete && cpus.first().is_some() {
        let count: u64 = cpus
            .ranges
            .iter()
            .map(|(first, last)| last - first + 1)
            .sum();
        result.cpu = u32::try_from(count).ok().map(|compute_units| CpuInfo {
            name: None,
            compute_units,
        });
        result.online_cpus = Some(cpus);
    }
    result.domains.sort_by_key(|domain| domain.numa_node);
    for domain in &mut result.domains {
        domain.caches.complete = result.caches.complete;
        domain.caches.records = result
            .caches
            .records
            .iter()
            .filter(|cache| {
                domain
                    .cpus
                    .as_ref()
                    .zip(cache.shared_cpus.as_ref())
                    .is_some_and(|(a, b)| a.intersects(b))
            })
            .cloned()
            .collect();
    }
    result
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn record(kind: u32, prefix: usize, groups: &[(u16, u64)], legacy: bool) -> Vec<u8> {
        let mut bytes = vec![0; 8 + prefix + groups.len() * 16];
        let length = u32::try_from(bytes.len()).unwrap();
        bytes[..4].copy_from_slice(&kind.to_le_bytes());
        bytes[4..8].copy_from_slice(&length.to_le_bytes());
        if !legacy {
            let count = u16::try_from(groups.len()).unwrap();
            bytes[8 + prefix - 2..8 + prefix].copy_from_slice(&count.to_le_bytes());
        }
        for (index, (group, mask)) in groups.iter().enumerate() {
            let offset = 8 + prefix + index * 16;
            bytes[offset..offset + 8].copy_from_slice(&mask.to_le_bytes());
            bytes[offset + 8..offset + 10].copy_from_slice(&group.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn processor_groups_numa_and_shared_caches_preserve_native_identity() {
        let mut bytes = record(0, 24, &[(0, u64::MAX)], false);
        bytes.extend(record(0, 24, &[(2, 3)], false));
        let mut numa = record(1, 24, &[(2, 3)], true);
        numa[8..12].copy_from_slice(&7_u32.to_le_bytes());
        bytes.extend(numa);
        let mut empty = record(1, 24, &[(0, 0)], false);
        empty[8..12].copy_from_slice(&9_u32.to_le_bytes());
        bytes.extend(empty);
        let mut cache = record(2, 32, &[(0, 1), (2, 3)], false);
        cache[8] = 3;
        cache[10..12].copy_from_slice(&64_u16.to_le_bytes());
        cache[12..16].copy_from_slice(&(8_u32 << 20).to_le_bytes());
        bytes.extend(cache);
        let snapshot = decode(&bytes);
        assert_eq!(
            snapshot.cpu,
            Some(CpuInfo {
                name: None,
                compute_units: 66
            })
        );
        let online = snapshot.online_cpus.as_ref().unwrap();
        assert!(online.contains(CpuId::windows(2, 1).unwrap()));
        assert!(!online.contains(CpuId::windows(1, 1).unwrap()));
        assert!(!online.contains(CpuId::linux(1)));
        assert!(snapshot.cpu_apic_ids.is_empty());
        assert_eq!(snapshot.domain(7).unwrap().caches.records.len(), 1);
        assert!(
            snapshot
                .domain(9)
                .unwrap()
                .cpus
                .as_ref()
                .unwrap()
                .first()
                .is_none()
        );
        assert!(snapshot.domain(9).unwrap().caches.records.is_empty());
        assert!(snapshot.domains_complete && snapshot.caches.complete);
    }

    #[test]
    fn legacy_cache_and_extended_numa_arrays_are_bounded() {
        let mut cache = record(2, 32, &[(3, 5)], true);
        cache[8] = 1;
        cache[10..12].copy_from_slice(&64_u16.to_le_bytes());
        cache[12..16].copy_from_slice(&32768_u32.to_le_bytes());
        let mut bytes = record(1, 24, &[(2, 1), (3, 5)], false);
        bytes.extend(&cache);
        let snapshot = decode(&bytes);
        assert_eq!(snapshot.domain(0).unwrap().caches.records.len(), 1);
        assert_eq!(
            snapshot.caches.records[0].first_shared_cpu(),
            CpuId::windows(3, 0)
        );
        for length in 1..cache.len() {
            let snapshot = decode(&cache[..length]);
            assert!(snapshot.caches.records.is_empty());
            assert!(!snapshot.caches.complete);
        }
        cache[38..40].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(!decode(&cache).caches.complete);
    }

    #[test]
    fn malformed_records_preserve_independent_observations() {
        let processor = record(0, 24, &[(0, 3)], false);
        let mut bytes = processor.clone();
        bytes.extend(record(2, 32, &[], false));
        let snapshot = decode(&bytes);
        assert_eq!(snapshot.cpu.unwrap().compute_units, 2);
        assert!(!snapshot.caches.complete);
        assert!(snapshot.domains_complete);
        for suffix in [
            vec![0; 4],
            [0_u32.to_le_bytes(), 0_u32.to_le_bytes()].concat(),
            [0_u32.to_le_bytes(), u32::MAX.to_le_bytes()].concat(),
        ] {
            let mut bytes = processor.clone();
            bytes.extend(suffix);
            let snapshot = decode(&bytes);
            assert!(snapshot.cpu.is_none() && snapshot.online_cpus.is_none());
            assert!(!snapshot.caches.complete && !snapshot.domains_complete);
        }
        let mut bytes = processor;
        bytes.extend(record(99, 0, &[], true));
        assert_eq!(decode(&bytes).cpu.unwrap().compute_units, 2);
    }
}
