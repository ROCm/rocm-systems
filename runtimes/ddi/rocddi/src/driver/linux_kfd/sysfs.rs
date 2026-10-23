// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive Linux topology discovery and validation.
//!
//! Discovery reads fixed-size sysfs records and opens no native execution
//! endpoint. Endpoint IDs select one node directly and are checked against its
//! current native identity before publication and again before activation.
//! Parsers reject incomplete, duplicate, overflowing, or internally inconsistent
//! properties rather than filling important device facts with defaults.
use super::{errno, error, native_error, sys};
use crate::driver::{DriverInstanceId, KfdAccessTarget};
use crate::memory::storage::Shared;
use crate::memory::storage::TryPush;
use crate::memory::{DeviceAccess, HostCacheability};
use crate::topology::{
    CacheInfo, CacheKind, CpuNodeInfo, Endpoint, EndpointKind, GpuInfo, GpuQueueCapabilities,
    MemoryBank, MemoryBankKind, MemoryLinkInfo, MemoryLinkType, MemoryLocation, MemoryLocationKind,
    NativeNodeId, NodeKind, PciInfo, SourceKind, TopologyCache, TopologySnapshot,
};
use crate::{Error, ErrorKind};
use std::fmt::{self, Write as _};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read as _};
use std::path::Path;

/// Stable KFD and DRM identity and capabilities for one topology node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KfdNode {
    /// Sysfs topology node number.
    pub node: u32,
    /// Native KFD GPU identifier used by ioctls.
    pub gpu_id: u32,
    /// DRM render node minor, when this node has one.
    pub render_minor: Option<u32>,
    /// Stable hardware identifier when reported by KFD.
    pub unique_id: Option<u64>,
    /// Endpoint identity used to detect topology changes.
    pub identity: [u8; 16],
    /// Queue and shader geometry qualified for this node.
    pub queues: KfdQueueProperties,
    /// Total local device memory in bytes.
    pub local_memory_bytes: u64,
    /// Host-visible local memory in bytes.
    pub public_memory_bytes: u64,
}

/// Queue and shader properties required to qualify KFD transports.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct KfdQueueProperties {
    /// Encoded GFX target reported by KFD.
    pub gfx_target: u32,
    /// Enabled compute-unit count.
    pub compute_units: u32,
    /// Maximum resident waves per compute unit.
    pub maximum_wave_count_per_compute_unit: u32,
    /// Maximum scratch waves per compute unit.
    pub maximum_scratch_wave_count_per_compute_unit: u32,
    /// Hardware lanes per wavefront.
    pub wavefront_size: u32,
    /// Number of shader XCCs exposed to this process.
    pub xcc_count: u32,
    /// Shader engines per XCC.
    pub shader_engine_count_per_xcc: u32,
    /// Native context-save allocation size.
    pub context_size: u32,
    /// Native control-stack allocation size.
    pub control_stack_size: u32,
    /// Number of ordinary SDMA engines.
    pub sdma_engines: u32,
    /// Number of SDMA engines for XGMI links.
    pub sdma_xgmi_engines: u32,
    /// Available global wave synchronization slots.
    pub gws_count: u32,
    /// Maximum native compute queues.
    pub compute_queues: u32,
    /// SDMA queue support validated against native properties.
    pub sdma_qualified: bool,
}

/// Bounded path builder for sysfs records discovered without allocation.
struct StackPath {
    /// UTF-8 path bytes written so far.
    bytes: [u8; 512],
    /// Length of the initialized byte prefix.
    len: usize,
}
impl StackPath {
    fn new(arguments: fmt::Arguments<'_>) -> Result<Self, Error> {
        let mut path = Self {
            bytes: [0; 512],
            len: 0,
        };
        path.write_fmt(arguments)
            .map_err(|_| error(ErrorKind::InvalidData, "sysfs path exceeds fixed bound"))?;
        Ok(path)
    }
    fn path(&self) -> &Path {
        // Every byte entered through fmt::Write and therefore is valid UTF-8.
        Path::new(std::str::from_utf8(&self.bytes[..self.len]).unwrap_or(""))
    }
}
impl fmt::Write for StackPath {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let end = self
            .len
            .checked_add(value.len())
            .filter(|end| *end <= self.bytes.len())
            .ok_or(fmt::Error)?;
        self.bytes[self.len..end].copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }
}

fn read<'a>(path: &Path, bytes: &'a mut [u8]) -> Result<&'a str, Error> {
    let mut file = File::open(path).map_err(|e| native_error("sysfs attribute open", e))?;
    let mut length = 0;
    while length < bytes.len() {
        let count = file
            .read(&mut bytes[length..])
            .map_err(|e| native_error("sysfs attribute read", e))?;
        if count == 0 {
            break;
        }
        length += count;
    }
    if length == bytes.len() {
        return Err(error(
            ErrorKind::InvalidData,
            "sysfs attribute exceeds fixed bound",
        ));
    }
    std::str::from_utf8(&bytes[..length])
        .map_err(|_| error(ErrorKind::InvalidData, "sysfs attribute is not UTF-8"))
}
fn scalar(arguments: fmt::Arguments<'_>) -> Result<u64, Error> {
    let path = StackPath::new(arguments)?;
    let mut bytes = [0; 128];
    parse_number(read(path.path(), &mut bytes)?.trim())
}
fn parse_number(text: &str) -> Result<u64, Error> {
    let result = if let Some(hex) = text.strip_prefix("0x") {
        u64::from_str_radix(hex, 16)
    } else {
        text.parse()
    };
    result.map_err(|_| error(ErrorKind::InvalidData, "invalid numeric sysfs property"))
}
/// Validated key-value lines from one native sysfs properties record.
struct Properties<'a>(&'a str);
impl<'a> Properties<'a> {
    fn new(text: &'a str) -> Result<Self, Error> {
        for (index, line) in text.lines().enumerate() {
            let mut parts = line.split_whitespace();
            let key = parts
                .next()
                .ok_or_else(|| error(ErrorKind::InvalidData, "empty sysfs property"))?;
            let value = parts
                .next()
                .ok_or_else(|| error(ErrorKind::InvalidData, "missing sysfs property value"))?;
            parse_number(value)?;
            if parts.next().is_some()
                || text
                    .lines()
                    .take(index)
                    .any(|line| line.split_whitespace().next() == Some(key))
            {
                return Err(error(
                    ErrorKind::InvalidData,
                    "duplicate or malformed sysfs property",
                ));
            }
        }
        Ok(Self(text))
    }
    fn optional(&self, key: &str) -> Result<Option<u64>, Error> {
        for line in self.0.lines() {
            let mut fields = line.split_whitespace();
            if fields.next() == Some(key) {
                return fields.next().map(parse_number).transpose();
            }
        }
        Ok(None)
    }
    fn required(&self, key: &str) -> Result<u64, Error> {
        self.optional(key)?.ok_or_else(|| {
            error(
                ErrorKind::InvalidData,
                "required native property is missing",
            )
        })
    }
    fn u32(&self, key: &str) -> Result<u32, Error> {
        u32::try_from(self.required(key)?)
            .map_err(|_| error(ErrorKind::InvalidData, "native property exceeds u32"))
    }
    fn optional_u32(&self, key: &str) -> Result<Option<u32>, Error> {
        self.optional(key)?
            .map(|v| {
                u32::try_from(v)
                    .map_err(|_| error(ErrorKind::InvalidData, "native property exceeds u32"))
            })
            .transpose()
    }
}

fn cpu_node_info(properties: &Properties<'_>) -> Result<Option<CpuNodeInfo>, Error> {
    let count = properties.optional_u32("cpu_cores_count")?.unwrap_or(0);
    let base = properties.optional_u32("cpu_core_id_base")?;
    Ok((count != 0).then_some(CpuNodeInfo {
        core_count: count,
        core_id_base: base,
        maximum_clock_mhz: properties.optional_u32("max_engine_clk_ccompute")?,
    }))
}

fn cache_properties(text: &str) -> Result<(CacheInfo, u32), Error> {
    let mut sibling_map = false;
    for (index, line) in text.lines().enumerate() {
        let mut parts = line.split_whitespace();
        let key = parts
            .next()
            .ok_or_else(|| error(ErrorKind::InvalidData, "empty cache property"))?;
        let value = parts
            .next()
            .ok_or_else(|| error(ErrorKind::InvalidData, "missing cache property value"))?;
        let duplicate = text
            .lines()
            .take(index)
            .any(|line| line.split_whitespace().next() == Some(key));
        if duplicate || parts.next().is_some() {
            return Err(error(
                ErrorKind::InvalidData,
                "duplicate or malformed cache property",
            ));
        }
        if key == "sibling_map" {
            if value.split(',').any(|bit| !matches!(bit, "0" | "1")) {
                return Err(error(
                    ErrorKind::InvalidData,
                    "invalid native cache sibling map",
                ));
            }
            sibling_map = true;
        } else {
            parse_number(value)?;
        }
    }
    if !sibling_map {
        return Err(error(
            ErrorKind::InvalidData,
            "required native cache property is missing",
        ));
    }
    let properties = Properties(text);
    let _ = properties.u32("processor_id_low")?;
    let line_size_bytes = properties.u32("cache_line_size")?;
    let _ = properties.u32("cache_lines_per_tag")?;
    let _ = properties.u32("association")?;
    let _ = properties.u32("latency")?;
    let native_kind = properties.u32("type")?;
    let kind = match (native_kind & 1 != 0, native_kind & 2 != 0) {
        (true, true) => CacheKind::Unified,
        (true, false) => CacheKind::Data,
        (false, true) => CacheKind::Instruction,
        (false, false) => CacheKind::Other,
    };
    // KFD reports cache capacity in KiB; driver facts use bytes.
    Ok((
        CacheInfo {
            level: properties.u32("level")?,
            size_bytes: u64::from(properties.u32("size")?) * 1024,
            line_size_bytes: (line_size_bytes != 0).then_some(line_size_bytes),
            kind,
            cpu_cache: native_kind & (1 << 2) != 0,
            compute_unit_cache: native_kind & (1 << 3) != 0,
        },
        properties
            .optional_u32("persisting_cache_size_max")?
            .unwrap_or(0),
    ))
}

/// Cache records and the largest reported persisting-L2 reservation.
struct DiscoveredCaches {
    /// Cache records ordered by native ordinal when any are present.
    records: Option<Shared<Vec<CacheInfo>>>,
    /// Maximum persisting L2 size reported by the device.
    persisting_l2_cache_size_max: u32,
}

fn caches(root: &str, node: u32, count: u32) -> Result<DiscoveredCaches, Error> {
    if count == 0 {
        return Ok(DiscoveredCaches {
            records: None,
            persisting_l2_cache_size_max: 0,
        });
    }
    let capacity = usize::try_from(count)
        .map_err(|_| error(ErrorKind::ResourceExhausted, "native cache count overflow"))?;
    let mut records = crate::memory::storage::try_vec_with_capacity(capacity)?;
    let path = StackPath::new(format_args!("{root}/nodes/{node}/caches"))?;
    sys::numeric_directories(path.path(), &mut |ordinal| {
        if ordinal >= count || records.iter().any(|(seen, _)| *seen == ordinal) {
            return Err(error(
                ErrorKind::InvalidData,
                "native cache ordinal is outside its declared range",
            ));
        }
        let path = StackPath::new(format_args!(
            "{root}/nodes/{node}/caches/{ordinal}/properties"
        ))?;
        let mut bytes = [0; 4096];
        records.try_push((ordinal, cache_properties(read(path.path(), &mut bytes)?)?))?;
        Ok(())
    })?;
    if records.len() != capacity {
        return Err(error(
            ErrorKind::InvalidData,
            "native cache count does not match its directory",
        ));
    }
    records.sort_unstable_by_key(|(ordinal, _)| *ordinal);
    if records
        .iter()
        .enumerate()
        .any(|(expected, (ordinal, _))| usize::try_from(*ordinal) != Ok(expected))
    {
        return Err(error(
            ErrorKind::InvalidData,
            "native cache ordinals are not contiguous",
        ));
    }
    let mut cache_info = crate::memory::storage::try_vec_with_capacity(capacity)?;
    let mut persisting_l2_cache_size_max = 0;
    for (_, (cache, persisting_cache_size_max)) in records {
        if cache.level == 2 && persisting_l2_cache_size_max == 0 {
            persisting_l2_cache_size_max = persisting_cache_size_max;
        }
        cache_info.try_push(cache)?;
    }
    Shared::new(cache_info)
        .map(|records| DiscoveredCaches {
            records: Some(records),
            persisting_l2_cache_size_max,
        })
        .map_err(Into::into)
}

/// One KFD bank report before it is associated with a source-qualified node.
#[derive(Clone, Copy)]
struct DiscoveredBank {
    ordinal: u32,
    kind: MemoryBankKind,
    size: u64,
    flags: u32,
    width: u32,
    clock: u32,
}

fn memory_bank_properties(text: &str, ordinal: u32) -> Result<DiscoveredBank, Error> {
    let properties = Properties::new(text)?;
    let heap = properties.u32("heap_type")?;
    let size = properties.required("size_in_bytes")?;
    let kind = match heap {
        0 => MemoryBankKind::System,
        1 => MemoryBankKind::FrameBufferPublic,
        2 => MemoryBankKind::FrameBufferPrivate,
        3 => MemoryBankKind::Gds,
        4 => MemoryBankKind::Lds,
        5 => MemoryBankKind::Scratch,
        6 => MemoryBankKind::DeviceSvm,
        7 => MemoryBankKind::Mmio,
        other => MemoryBankKind::Other(other),
    };
    Ok(DiscoveredBank {
        ordinal,
        kind,
        size,
        flags: properties.optional_u32("flags")?.unwrap_or(0),
        width: properties.optional_u32("width")?.unwrap_or(0),
        clock: properties.optional_u32("mem_clk_max")?.unwrap_or(0),
    })
}

fn memory_banks(root: &str, node: u32, count: u32) -> Result<Vec<DiscoveredBank>, Error> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let capacity = usize::try_from(count).map_err(|_| {
        error(
            ErrorKind::ResourceExhausted,
            "native memory-bank count overflow",
        )
    })?;
    let mut records = crate::memory::storage::try_vec_with_capacity(capacity)?;
    let path = StackPath::new(format_args!("{root}/nodes/{node}/mem_banks"))?;
    sys::numeric_directories(path.path(), &mut |ordinal| {
        if ordinal >= count || records.iter().any(|(seen, _)| *seen == ordinal) {
            return Err(error(
                ErrorKind::InvalidData,
                "native memory-bank ordinal is outside its declared range",
            ));
        }
        let path = StackPath::new(format_args!(
            "{root}/nodes/{node}/mem_banks/{ordinal}/properties"
        ))?;
        let mut bytes = [0; 4096];
        records.try_push((
            ordinal,
            memory_bank_properties(read(path.path(), &mut bytes)?, ordinal)?,
        ))?;
        Ok(())
    })?;
    if records.len() != capacity {
        return Err(error(
            ErrorKind::InvalidData,
            "native memory-bank count does not match its directory",
        ));
    }
    records.sort_unstable_by_key(|(ordinal, _)| *ordinal);
    for (expected, (ordinal, bank)) in records.iter().enumerate() {
        if usize::try_from(*ordinal) != Ok(expected) {
            return Err(error(
                ErrorKind::InvalidData,
                "native memory-bank ordinals are not contiguous",
            ));
        }
        debug_assert_eq!(*ordinal, bank.ordinal);
    }
    Ok(records.into_iter().map(|(_, bank)| bank).collect())
}

fn memory_bus(banks: &[DiscoveredBank]) -> (u32, u32) {
    banks
        .iter()
        .rfind(|bank| {
            bank.size != 0
                && matches!(
                    bank.kind,
                    MemoryBankKind::FrameBufferPublic | MemoryBankKind::FrameBufferPrivate
                )
        })
        .map_or((0, 0), |bank| (bank.width, bank.clock))
}

struct DiscoveredEndpoint {
    endpoint: Endpoint,
    caches: Option<Shared<Vec<CacheInfo>>>,
    links: Option<Shared<Vec<KfdReportedLink>>>,
    cpu: Option<CpuNodeInfo>,
    banks: Vec<DiscoveredBank>,
}

/// One directed KFD node report, before resolution into snapshot node IDs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct KfdReportedLink {
    source: KfdAccessTarget,
    target: KfdAccessTarget,
    source_is_host: bool,
    target_is_host: bool,
    info: MemoryLinkInfo,
    peer_dma_allowed: bool,
}

struct DiscoveredNode {
    node: u32,
    gpu_id: u32,
    endpoint: Option<Endpoint>,
    caches: Option<Shared<Vec<CacheInfo>>>,
    links: Option<Shared<Vec<KfdReportedLink>>>,
    cpu: Option<CpuNodeInfo>,
    banks: Vec<DiscoveredBank>,
}

struct DiscoveredTopology {
    generation: Option<u64>,
    nodes: Vec<DiscoveredNode>,
}

fn read_all(
    root: &str,
    drm: &str,
    driver_instance: DriverInstanceId,
    include_host_facts: bool,
) -> Result<DiscoveredTopology, Error> {
    for _ in 0..3 {
        let before = match scalar(format_args!("{root}/generation_id")) {
            Ok(value) => value,
            Err(e) if e.native_error_code() == Some(errno::ENOENT) => {
                return Ok(DiscoveredTopology {
                    generation: None,
                    nodes: Vec::new(),
                });
            }
            Err(e) => return Err(e),
        };
        let mut records = Vec::new();
        let nodes = StackPath::new(format_args!("{root}/nodes"))?;
        let result = sys::numeric_directories(nodes.path(), &mut |node| {
            let gpu_id = u32::try_from(scalar(format_args!("{root}/nodes/{node}/gpu_id"))?)
                .map_err(|_| error(ErrorKind::InvalidData, "GPU ID exceeds u32"))?;
            let record = if gpu_id == 0 && include_host_facts {
                read_host_node(root, node)?
            } else if gpu_id == 0 {
                DiscoveredNode {
                    node,
                    gpu_id,
                    endpoint: None,
                    caches: None,
                    links: None,
                    cpu: None,
                    banks: Vec::new(),
                }
            } else {
                let endpoint = read_node(root, drm, driver_instance, node)?.ok_or_else(|| {
                    error(ErrorKind::InvalidData, "native GPU node lost its GPU ID")
                })?;
                DiscoveredNode {
                    node,
                    gpu_id,
                    endpoint: Some(endpoint.endpoint),
                    caches: endpoint.caches,
                    links: endpoint.links,
                    cpu: endpoint.cpu,
                    banks: endpoint.banks,
                }
            };
            records.try_push(record)?;
            Ok(())
        });
        let after = scalar(format_args!("{root}/generation_id"))?;
        if before != after {
            continue;
        }
        result?;
        records.sort_unstable_by_key(|record| record.node);
        return Ok(DiscoveredTopology {
            generation: Some(before),
            nodes: records,
        });
    }
    Err(error(
        ErrorKind::ConcurrentModification,
        "native topology changed during every read",
    ))
}

pub(super) fn enumerate(
    root: &str,
    drm: &str,
    driver_instance: DriverInstanceId,
    visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
) -> Result<(), Error> {
    let discovered = read_all(root, drm, driver_instance, false)?;
    for record in discovered.nodes {
        if let Some(endpoint) = record.endpoint {
            visitor(endpoint)?;
        }
    }
    Ok(())
}

pub(super) fn contribute_topology(
    root: &str,
    drm: &str,
    driver_instance: DriverInstanceId,
    snapshot: &mut TopologySnapshot,
) -> Result<(), Error> {
    let discovered = read_all(root, drm, driver_instance, true)?;
    let source = snapshot.add_source(
        SourceKind::LinuxKfd,
        discovered.generation,
        true,
        Some(driver_instance),
    )?;
    let mut nodes = Vec::new();
    for record in &discovered.nodes {
        let id = snapshot.add_node(
            source,
            NativeNodeId::LinuxKfd(record.node),
            if record.gpu_id == 0 {
                NodeKind::KfdHost
            } else {
                NodeKind::KfdGpu
            },
        )?;
        nodes.try_push((record.node, record.gpu_id, id))?;
    }
    for record in discovered.nodes {
        let node = nodes
            .iter()
            .find(|(native, _, _)| *native == record.node)
            .map(|(_, _, id)| *id)
            .ok_or_else(|| error(ErrorKind::InvalidData, "KFD endpoint has no topology node"))?;
        if let Some(cpu) = record.cpu {
            snapshot.set_cpu_node(node, cpu)?;
        }
        for bank in &record.banks {
            snapshot.add_memory_bank(MemoryBank {
                node,
                native_ordinal: bank.ordinal,
                kind: bank.kind,
                capacity_bytes: bank.size,
                native_flags: bank.flags,
                width_bits: (bank.width != 0).then_some(bank.width),
                maximum_clock_mhz: (bank.clock != 0).then_some(bank.clock),
            })?;
        }
        if let Some(caches) = &record.caches {
            for cache in caches.iter() {
                snapshot
                    .caches
                    .try_push(TopologyCache { node, info: *cache })?;
            }
        }
        if let Some(links) = &record.links {
            for link in links.iter() {
                let resolve = |key: KfdAccessTarget| {
                    nodes
                        .iter()
                        .find(|(native, gpu, _)| *native == key.node && *gpu == key.gpu_id)
                        .map(|(_, _, id)| *id)
                        .ok_or_else(|| {
                            error(ErrorKind::InvalidData, "KFD route names an unknown node")
                        })
                };
                snapshot.add_route(resolve(link.source)?, resolve(link.target)?, link.info)?;
            }
        }
        if let Some(endpoint) = record.endpoint {
            let local = endpoint.local_memory_bytes;
            let visible = endpoint.host_visible_local_memory_bytes;
            snapshot.add_endpoint(endpoint, node)?;
            if local != 0 {
                snapshot.add_memory_location(MemoryLocation {
                    node,
                    kind: MemoryLocationKind::DeviceLocal,
                    capacity_bytes: Some(local),
                    host_visible_bytes: Some(visible),
                })?;
            }
        }
    }
    Ok(())
}

pub(super) fn open_endpoint(
    root: &str,
    drm: &str,
    driver_instance: DriverInstanceId,
    id: [u8; 16],
) -> Result<Endpoint, Error> {
    let node = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
    let before = scalar(format_args!("{root}/generation_id"))?;
    let endpoint = read_node(root, drm, driver_instance, node)?
        .ok_or_else(|| error(ErrorKind::DeviceLost, "native GPU endpoint disappeared"))?;
    let after = scalar(format_args!("{root}/generation_id"))?;
    if before != after {
        return Err(error(
            ErrorKind::ConcurrentModification,
            "native endpoint changed during read",
        ));
    }
    if endpoint.endpoint.id != id {
        return Err(error(
            ErrorKind::DeviceLost,
            "native endpoint identity changed",
        ));
    }
    Ok(endpoint.endpoint)
}

/// Distinguishes an unrecognized native ID from a failure while reading an
/// existing node. Only the node directory and its identity fields participate
/// in the probe; missing cache, link, or DRM metadata remains an open error.
pub(super) fn probe_endpoint(
    root: &str,
    drm: &str,
    driver_instance: DriverInstanceId,
    id: [u8; 16],
) -> Result<Option<Endpoint>, Error> {
    let node = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
    let requested_gpu = u32::from_le_bytes([id[4], id[5], id[6], id[7]]);
    let requested_tail = u64::from_le_bytes(id[8..].try_into().map_err(|_| {
        error(
            ErrorKind::InvalidData,
            "native endpoint ID has no identity tail",
        )
    })?);
    let before = match scalar(format_args!("{root}/generation_id")) {
        Ok(generation) => generation,
        Err(error) if error.native_error_code() == Some(errno::ENOENT) => return Ok(None),
        Err(error) => return Err(error),
    };
    let path = StackPath::new(format_args!("{root}/nodes/{node}"))?;
    let exists = match fs::metadata(path.path()) {
        Ok(metadata) if metadata.is_dir() => true,
        Ok(_) => {
            return Err(error(
                ErrorKind::InvalidData,
                "native node is not a directory",
            ));
        }
        Err(source) if source.kind() == io::ErrorKind::NotFound => false,
        Err(source) => return Err(native_error("native node lookup", source)),
    };
    let identity = if exists {
        let gpu_id = u32::try_from(scalar(format_args!("{root}/nodes/{node}/gpu_id"))?)
            .map_err(|_| error(ErrorKind::InvalidData, "GPU ID exceeds u32"))?;
        if gpu_id == 0 || gpu_id != requested_gpu {
            None
        } else {
            let path = StackPath::new(format_args!("{root}/nodes/{node}/properties"))?;
            let mut bytes = [0; 16384];
            let properties = Properties::new(read(path.path(), &mut bytes)?)?;
            let unique_id = properties
                .optional("unique_id")?
                .filter(|value| *value != 0);
            let domain = properties.u32("domain")?;
            let location = properties.u32("location_id")?;
            Some(unique_id.unwrap_or((u64::from(domain) << 32) | u64::from(location)))
        }
    } else {
        None
    };
    let after = scalar(format_args!("{root}/generation_id"))?;
    if before != after {
        return Err(error(
            ErrorKind::ConcurrentModification,
            "native endpoint changed during identity probe",
        ));
    }
    if identity.is_none() {
        return Ok(None);
    }
    if identity != Some(requested_tail) {
        return Err(error(
            ErrorKind::DeviceLost,
            "native endpoint identity changed",
        ));
    }
    open_endpoint(root, drm, driver_instance, id).map(Some)
}

fn kfd_link(
    node: u32,
    gpu_id: u32,
    target: u32,
    target_gpu_id: u32,
    info: MemoryLinkInfo,
    peer_dma_allowed: bool,
) -> KfdReportedLink {
    KfdReportedLink {
        source: KfdAccessTarget { node, gpu_id },
        target: KfdAccessTarget {
            node: target,
            gpu_id: target_gpu_id,
        },
        source_is_host: gpu_id == 0,
        target_is_host: target_gpu_id == 0,
        info,
        peer_dma_allowed,
    }
}

fn read_host_node(root: &str, node: u32) -> Result<DiscoveredNode, Error> {
    let path = StackPath::new(format_args!("{root}/nodes/{node}/properties"))?;
    let mut bytes = [0; 16384];
    let properties = Properties::new(read(path.path(), &mut bytes)?)?;
    let cpu = cpu_node_info(&properties)?;
    let banks = memory_banks(
        root,
        node,
        properties.optional_u32("mem_banks_count")?.unwrap_or(0),
    )?;
    let caches = caches(
        root,
        node,
        properties.optional_u32("caches_count")?.unwrap_or(0),
    )?
    .records;
    let links = memory_links(
        root,
        node,
        0,
        properties.optional_u32("io_links_count")?.unwrap_or(0),
        properties.optional_u32("p2p_links_count")?.unwrap_or(0),
    )?;
    Ok(DiscoveredNode {
        node,
        gpu_id: 0,
        endpoint: None,
        caches,
        links,
        cpu,
        banks,
    })
}

fn node_memory_links(
    root: &str,
    node: u32,
    gpu_id: u32,
    io_link_count: u32,
    p2p_link_count: u32,
) -> Result<Vec<KfdReportedLink>, Error> {
    const OVERRIDE: u32 = 1;
    const NON_COHERENT: u32 = 1 << 1;
    const NO_ATOMICS_32: u32 = 1 << 2;
    const NO_ATOMICS_64: u32 = 1 << 3;
    const NO_PEER_TO_PEER_DMA: u32 = 1 << 4;
    let capacity = usize::try_from(io_link_count)
        .ok()
        .and_then(|count| count.checked_add(p2p_link_count as usize))
        .ok_or_else(|| error(ErrorKind::ResourceExhausted, "native link count overflow"))?;
    let mut links: Vec<KfdReportedLink> = crate::memory::storage::try_vec_with_capacity(capacity)?;
    for (directory, expected) in [("io_links", io_link_count), ("p2p_links", p2p_link_count)] {
        if expected == 0 {
            continue;
        }
        let path = StackPath::new(format_args!("{root}/nodes/{node}/{directory}"))?;
        let mut seen = 0u32;
        sys::numeric_directories(path.path(), &mut |link| {
            seen = seen
                .checked_add(1)
                .ok_or_else(|| error(ErrorKind::ResourceExhausted, "native link count overflow"))?;
            let path = StackPath::new(format_args!(
                "{root}/nodes/{node}/{directory}/{link}/properties"
            ))?;
            let mut bytes = [0; 4096];
            let properties = Properties::new(read(path.path(), &mut bytes)?)?;
            let kind = properties.u32("type")?;
            let from = properties.u32("node_from")?;
            let to = properties.u32("node_to")?;
            let flags = properties.u32("flags")?;
            if from != node {
                return Err(error(
                    ErrorKind::InvalidData,
                    "native link source does not match its node",
                ));
            }
            let target_gpu_id = u32::try_from(scalar(format_args!("{root}/nodes/{to}/gpu_id"))?)
                .map_err(|_| error(ErrorKind::InvalidData, "peer GPU ID exceeds u32"))?;
            let (kind, default_capabilities) = match kind {
                1 => (MemoryLinkType::HyperTransport, true),
                2 => (MemoryLinkType::Pcie, true),
                5 => (MemoryLinkType::Qpi, true),
                9 => (MemoryLinkType::Infiniband, false),
                11 => (MemoryLinkType::Xgmi, true),
                _ => (MemoryLinkType::Unknown, false),
            };
            let overridden = flags & OVERRIDE != 0;
            let peer_dma_allowed = !overridden || flags & NO_PEER_TO_PEER_DMA == 0;
            let info = MemoryLinkInfo {
                minimum_latency: properties.optional_u32("min_latency")?.unwrap_or(0),
                maximum_latency: properties.optional_u32("max_latency")?.unwrap_or(0),
                minimum_bandwidth: properties.optional_u32("min_bandwidth")?.unwrap_or(0),
                maximum_bandwidth: properties.optional_u32("max_bandwidth")?.unwrap_or(0),
                atomic_support_32bit: if overridden {
                    flags & NO_ATOMICS_32 == 0
                } else {
                    default_capabilities
                },
                atomic_support_64bit: if overridden {
                    flags & NO_ATOMICS_64 == 0
                } else {
                    default_capabilities
                },
                coherent_support: if overridden {
                    flags & NON_COHERENT == 0
                } else {
                    default_capabilities
                },
                kind,
                numa_distance: properties.optional_u32("weight")?.unwrap_or(0),
            };
            let link = kfd_link(node, gpu_id, to, target_gpu_id, info, peer_dma_allowed);
            if let Some(existing) = links
                .iter_mut()
                .find(|existing| existing.source == link.source && existing.target == link.target)
            {
                *existing = link;
            } else {
                links.try_push(link)?;
            }
            Ok(())
        })?;
        if seen != expected {
            return Err(error(
                ErrorKind::InvalidData,
                "native link count does not match its directory",
            ));
        }
    }
    Ok(links)
}

fn memory_links(
    root: &str,
    node: u32,
    gpu_id: u32,
    io_link_count: u32,
    p2p_link_count: u32,
) -> Result<Option<Shared<Vec<KfdReportedLink>>>, Error> {
    let links = node_memory_links(root, node, gpu_id, io_link_count, p2p_link_count)?;
    if links.is_empty() {
        Ok(None)
    } else {
        Shared::new(links).map(Some).map_err(Into::into)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "all immutable native facts share one identity and generation bracket"
)]
fn read_node(
    root: &str,
    drm: &str,
    driver_instance: DriverInstanceId,
    node: u32,
) -> Result<Option<DiscoveredEndpoint>, Error> {
    let gpu_id = u32::try_from(scalar(format_args!("{root}/nodes/{node}/gpu_id"))?)
        .map_err(|_| error(ErrorKind::InvalidData, "GPU ID exceeds u32"))?;
    if gpu_id == 0 {
        return Ok(None);
    }
    let path = StackPath::new(format_args!("{root}/nodes/{node}/properties"))?;
    let mut bytes = [0; 16384];
    let p = Properties::new(read(path.path(), &mut bytes)?)?;
    let render = p.u32("drm_render_minor")?;
    let gfx = p.u32("gfx_target_version")?;
    let capability = p.u32("capability")?;
    let simds = p.u32("simd_count")?;
    let per_cu = p.u32("simd_per_cu")?;
    let xcc = p.u32("num_xcc")?;
    let arrays = p.u32("array_count")?;
    let arrays_per_engine = p.u32("simd_arrays_per_engine")?;
    let arrays_per_xcc = arrays_per_engine
        .checked_mul(xcc)
        .filter(|n| *n != 0)
        .ok_or_else(|| error(ErrorKind::InvalidData, "invalid native array geometry"))?;
    if per_cu == 0
        || simds == 0
        || simds % per_cu != 0
        || arrays == 0
        || arrays % arrays_per_xcc != 0
        || gfx == 0
    {
        return Err(error(
            ErrorKind::InvalidData,
            "invalid native compute geometry",
        ));
    }
    let waves = p
        .u32("max_waves_per_simd")?
        .checked_mul(per_cu)
        .filter(|n| *n != 0)
        .ok_or_else(|| error(ErrorKind::InvalidData, "invalid native wave geometry"))?;
    let scratch = p.u32("max_slots_scratch_cu")?;
    let lds = p.u32("lds_size_in_kb")?;
    let wave_size = p.u32("wave_front_size")?;
    if scratch == 0 || lds == 0 || !matches!(wave_size, 32 | 64) {
        return Err(error(
            ErrorKind::InvalidData,
            "invalid native compute limits",
        ));
    }
    let location = p.u32("location_id")?;
    let domain = p.u32("domain")?;
    let vendor = p.u32("vendor_id")?;
    let device = p.u32("device_id")?;
    if vendor != 0x1002 || device > u32::from(u16::MAX) || location > 65535 {
        return Err(error(ErrorKind::InvalidData, "invalid PCI identity"));
    }
    let unique_id = p.optional("unique_id")?.filter(|v| *v != 0);
    let hive_id = p.optional("hive_id")?.unwrap_or(0);
    let family_id = p.optional_u32("family_id")?.unwrap_or(0);
    let maximum_engine_clock_mhz = p.optional_u32("max_engine_clk_fcompute")?.unwrap_or(0);
    let banks = memory_banks(root, node, p.optional_u32("mem_banks_count")?.unwrap_or(0))?;
    let (memory_bus_width_bits, maximum_memory_clock_mhz) = memory_bus(&banks);
    let cpu = cpu_node_info(&p)?;
    let sdma_xgmi_engines = p.optional_u32("num_sdma_xgmi_engines")?.unwrap_or(0);
    let gws_count = p.optional_u32("num_gws")?.unwrap_or(0);
    let io_link_count = p.optional_u32("io_links_count")?.unwrap_or(0);
    let p2p_link_count = p.optional_u32("p2p_links_count")?.unwrap_or(0);
    let DiscoveredCaches {
        records: caches,
        persisting_l2_cache_size_max,
    } = caches(root, node, p.optional_u32("caches_count")?.unwrap_or(0))?;
    let memory_links = memory_links(root, node, gpu_id, io_link_count, p2p_link_count)?;
    let mut access_targets = Vec::new();
    if let Some(links) = &memory_links {
        for link in links.iter() {
            if link.source.node == node
                && link.source.gpu_id == gpu_id
                && !link.source_is_host
                && !link.target_is_host
                && link.peer_dma_allowed
                && link.info.hop_count() != 0
                && !access_targets.contains(&link.target)
            {
                access_targets.try_push(link.target)?;
            }
        }
    }
    let local_access_targets = if access_targets.is_empty() {
        None
    } else {
        Some(Shared::new(access_targets)?)
    };
    let identity_tail = unique_id.unwrap_or((u64::from(domain) << 32) | u64::from(location));
    let mut id = [0; 16];
    id[..4].copy_from_slice(&node.to_le_bytes());
    id[4..8].copy_from_slice(&gpu_id.to_le_bytes());
    id[8..].copy_from_slice(&identity_tail.to_le_bytes());
    let (bits, integrated) = target_memory(gfx, device).ok_or_else(|| {
        error(
            ErrorKind::Unsupported,
            "GPU target memory envelope is not qualified",
        )
    })?;
    let context = p.optional_u32("cwsr_size")?;
    let stack = p.optional_u32("ctl_stack_size")?;
    if context.is_some() != stack.is_some() {
        return Err(error(
            ErrorKind::InvalidData,
            "incomplete KFD queue context metadata",
        ));
    }
    let sdma = p.optional_u32("num_sdma_engines")?;
    let sdma_queues = p.optional_u32("num_sdma_queues_per_engine")?;
    if sdma.is_some() != sdma_queues.is_some() {
        return Err(error(
            ErrorKind::InvalidData,
            "incomplete native SDMA metadata",
        ));
    }
    let sdma_ip = sdma_version(drm, render)?;
    // This transport publishes monotonic 64-bit byte indices and 64-bit
    // doorbells across the supported GFX9–GFX12 native targets. GFX9 uses
    // ROCr's SDMA v4 packet path, so its IP version must match that dialect.
    // Commands remain target-specific and caller-owned; a discovered SDMA IP
    // does not change any frontend's advertised transport revision.
    let gfx9_sdma = matches!(gfx, 90008 | 90010 | 90402 | 90500)
        && sdma_ip.is_some_and(|(major, _, _)| major == 4);
    let sdma_qualified = (gfx9_sdma || (100_100..=120_001).contains(&gfx) || gfx == 120_500)
        && sdma.unwrap_or(0) != 0
        && sdma_queues.unwrap_or(0) != 0;

    let total = scalar(format_args!(
        "{drm}/renderD{render}/device/mem_info_vram_total"
    ))?;
    let visible = scalar(format_args!(
        "{drm}/renderD{render}/device/mem_info_vis_vram_total"
    ))?;
    if visible > total {
        return Err(error(
            ErrorKind::InvalidData,
            "visible VRAM exceeds total VRAM",
        ));
    }
    let local = if integrated { 0 } else { total };
    let public = if integrated { 0 } else { visible };
    let queues = KfdQueueProperties {
        gfx_target: gfx,
        compute_units: simds / per_cu,
        maximum_wave_count_per_compute_unit: waves,
        maximum_scratch_wave_count_per_compute_unit: scratch,
        wavefront_size: wave_size,
        xcc_count: xcc,
        shader_engine_count_per_xcc: arrays / arrays_per_xcc,
        context_size: context.unwrap_or(0),
        control_stack_size: stack.unwrap_or(0),
        sdma_engines: sdma.unwrap_or(0),
        sdma_xgmi_engines,
        gws_count,
        compute_queues: p.u32("num_cp_queues")?,
        sdma_qualified,
    };
    let mut name = [0; 128];
    let name_path = StackPath::new(format_args!("{root}/nodes/{node}/name"))?;
    let mut name_bytes = [0; 128];
    let text = read(name_path.path(), &mut name_bytes)?.trim();
    if text.as_bytes().contains(&0) {
        return Err(error(
            ErrorKind::InvalidData,
            "embedded NUL in native GPU name",
        ));
    }
    name[..text.len()].copy_from_slice(text.as_bytes());
    let native = KfdNode {
        node,
        gpu_id,
        render_minor: Some(render),
        unique_id,
        identity: id,
        queues,
        local_memory_bytes: local,
        public_memory_bytes: public,
    };
    let pci_number = |attribute| -> Result<u32, Error> {
        u32::try_from(scalar(format_args!(
            "{drm}/renderD{render}/device/{attribute}"
        ))?)
        .map_err(|_| error(ErrorKind::InvalidData, "PCI property exceeds u32"))
    };
    let pci = PciInfo {
        domain,
        bus: location >> 8,
        device: (location >> 3) & 31,
        function: location & 7,
        vendor_id: vendor,
        device_id: device,
        subsystem_vendor_id: pci_number("subsystem_vendor")?,
        subsystem_device_id: pci_number("subsystem_device")?,
        revision_id: pci_number("revision")?,
    };
    if pci_number("vendor")? != vendor || pci_number("device")? != device {
        return Err(error(
            ErrorKind::DeviceLost,
            "render endpoint no longer matches topology",
        ));
    }
    let supports_aql = super::queue::supports_aql(&native);
    let supports_pm4 = super::queue::supports_pm4(&native);
    let endpoint = Endpoint {
        id,
        name,
        product_name: None,
        pci: Some(pci),
        kind: EndpointKind::Gpu {
            info: GpuInfo {
                native_gpu_id: gpu_id,
                gfx_major: gfx / 10000,
                gfx_minor: (gfx / 100) % 100,
                gfx_stepping: gfx % 100,
                unique_id,
                hive_id,
                asic_family_id: family_id,
                drm_asic_family_id: None,
                counter_frequency_hz: None,
                asic_revision: (capability & 0x03c0_0000) >> 22,
                maximum_engine_clock_mhz,
                memory_bus_width_bits,
                maximum_memory_clock_mhz,
                wavefront_size: wave_size,
                compute_unit_count: simds / per_cu,
                simd_count_per_compute_unit: per_cu,
                maximum_wave_count_per_compute_unit: waves,
                maximum_address_watch_point_count: 1 << ((capability >> 8) & 0xf),
                maximum_scratch_wave_count_per_compute_unit: scratch,
                local_data_share_byte_length: u64::from(lds) * 1024,
                xcc_count: xcc,
                shader_engine_count_per_xcc: arrays / arrays_per_xcc,
                shader_array_count_per_engine: arrays_per_engine,
                sdma_xgmi_engine_count: sdma_xgmi_engines,
                gws_count,
                iommu_v2_supported: capability & (1 << 1) != 0,
                coherent_host_access: capability & (1 << 28) != 0,
                sram_ecc_enabled: capability & 0x0400_0000 != 0,
                packet_processor_firmware_version: p.optional_u32("fw_version")?.unwrap_or(0)
                    & 0x3ff,
                sdma_firmware_version: p.optional_u32("sdma_fw_version")?.unwrap_or(0) & 0x3ff,
                persisting_l2_cache_size_max,
                fabric_handles_supported: StackPath::new(format_args!(
                    "{drm}/renderD{render}/device/ualink"
                ))
                .is_ok_and(|path| File::open(path.path()).is_ok()),
                queues: GpuQueueCapabilities {
                    aql: supports_aql,
                    // GFX12 shares the AQL system-scope cache protocol.
                    aql_system_cache_control: cfg!(target_arch = "x86_64")
                        && supports_aql
                        && matches!(gfx, 120_000 | 120_001 | 120_500),
                    pm4: supports_pm4,
                    kernel_pm4: supports_pm4,
                    // PM4 uses the GFX12 ACQUIRE_MEM GCR encoding.
                    pm4_system_cache_control: supports_pm4,
                    sdma: sdma_qualified,
                    kernel_sdma: cfg!(target_arch = "x86_64")
                        && sdma_qualified
                        && ((matches!(gfx, 120_000 | 120_001) && sdma_ip == Some((7, 0, 1)))
                            || (gfx == 120_500 && sdma_ip == Some((7, 1, 0)))),
                    // USER_GCR requires the matching target-specific SDMA packet format.
                    sdma_system_cache_control: cfg!(target_arch = "x86_64")
                        && sdma_qualified
                        && ((matches!(gfx, 120_000 | 120_001) && sdma_ip == Some((7, 0, 1)))
                            || (gfx == 120_500 && sdma_ip == Some((7, 1, 0)))),
                    sdma_engine_count: if sdma_qualified { sdma.unwrap_or(0) } else { 0 },
                },
            },
        },
        local_memory_bytes: local,
        host_visible_local_memory_bytes: public,
        host_local_cacheability: (cfg!(target_arch = "x86_64")
            && public != 0
            && ((100_100..=120_001).contains(&gfx) || gfx == 120_500))
            .then_some(HostCacheability::WriteCombined),
        allocation_granularity: 4096,
        address_bit_count: bits,
        minimum_address: 65536,
        maximum_address: (1u64 << bits) - 1,
        supported_permissions: DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        driver_instance,
        selector: crate::driver::EndpointSelector::LinuxKfd {
            native,
            access_targets: local_access_targets,
        },
    };
    Ok(Some(DiscoveredEndpoint {
        endpoint,
        caches,
        links: memory_links,
        cpu,
        banks,
    }))
}

/// KFD-qualified passive access evidence. A native mapping still decides
/// whether a particular allocation can actually be made accessible.
pub(super) fn can_access_local_memory(consumer: &Endpoint, owner: &Endpoint) -> bool {
    let (
        crate::driver::EndpointSelector::LinuxKfd {
            native: from,
            access_targets,
        },
        crate::driver::EndpointSelector::LinuxKfd { native: to, .. },
    ) = (&consumer.selector, &owner.selector)
    else {
        return false;
    };
    consumer.driver_instance == owner.driver_instance
        && ((from.node == to.node && from.gpu_id == to.gpu_id)
            || access_targets.as_ref().is_some_and(|targets| {
                targets.contains(&KfdAccessTarget {
                    node: to.node,
                    gpu_id: to.gpu_id,
                })
            }))
}

fn sdma_version(drm: &str, render: u32) -> Result<Option<(u32, u32, u32)>, Error> {
    let path = StackPath::new(format_args!(
        "{drm}/renderD{render}/device/ip_discovery/die/0/42/0"
    ))?;
    match File::open(path.path()) {
        Ok(_) => (),
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(native_error("SDMA metadata open", e)),
    }
    let mut version = [0u32; 3];
    for (i, name) in ["major", "minor", "revision"].into_iter().enumerate() {
        let value = scalar(format_args!(
            "{drm}/renderD{render}/device/ip_discovery/die/0/42/0/{name}"
        ))?;
        version[i] = u32::try_from(value)
            .ok()
            .filter(|v| *v <= 255)
            .ok_or_else(|| error(ErrorKind::InvalidData, "invalid SDMA IP version"))?;
    }
    Ok(Some((version[0], version[1], version[2])))
}

fn target_memory(gfx: u32, device: u32) -> Option<(u32, bool)> {
    let result = match gfx {
        70000 | 80001 => (40, true),
        70001 | 80002 | 80003 => (40, false),
        90002
        | 90012
        | 100_103
        | 100_303
        | 100_305
        | 100_306
        | 110_003
        | 110_500..=110_504
        | 110_700
        | 110_701 => (47, true),
        90000
        | 90004
        | 90006
        | 90008
        | 90010
        | 90500
        | 100_100..=100_102
        | 100_300..=100_302
        | 100_304
        | 110_000..=110_002
        | 120_000
        | 120_001 => (47, false),
        90402 if device == 0x74a0 => (47, true),
        90402
            if matches!(
                device,
                0x74a1 | 0x74a2 | 0x74a5 | 0x74a8 | 0x74a9 | 0x74b5 | 0x74b6 | 0x74b9 | 0x74bd
            ) =>
        {
            (47, false)
        }
        120_500 => (56, false),
        _ => return None,
    };
    Some(result)
}

pub(super) fn open_render(minor: u32) -> io::Result<File> {
    let path = StackPath::new(format_args!("/dev/dri/renderD{minor}"))
        .map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    OpenOptions::new().read(true).write(true).open(path.path())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn enumerate(
        root: &str,
        drm: &str,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        super::enumerate(root, drm, DriverInstanceId::TEST, visitor)
    }

    fn open_endpoint(root: &str, drm: &str, id: [u8; 16]) -> Result<Endpoint, Error> {
        super::open_endpoint(root, drm, DriverInstanceId::TEST, id)
    }

    fn probe_endpoint(root: &str, drm: &str, id: [u8; 16]) -> Result<Option<Endpoint>, Error> {
        super::probe_endpoint(root, drm, DriverInstanceId::TEST, id)
    }

    fn topology(root: &str, drm: &str) -> TopologySnapshot {
        let mut snapshot = TopologySnapshot::new(crate::topology::host::snapshot()).unwrap();
        super::contribute_topology(root, drm, DriverInstanceId::TEST, &mut snapshot).unwrap();
        snapshot
    }

    static NEXT: AtomicU32 = AtomicU32::new(0);
    struct Fixture(std::path::PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "rocddi-passive-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            std::fs::create_dir_all(root.join("topology/nodes/1")).unwrap();
            std::fs::create_dir_all(root.join("drm/renderD128/device")).unwrap();
            std::fs::write(root.join("topology/generation_id"), "1\n").unwrap();
            std::fs::write(root.join("topology/nodes/1/gpu_id"), "42\n").unwrap();
            std::fs::write(root.join("topology/nodes/1/name"), "test GPU\n").unwrap();
            std::fs::write(
                root.join("topology/nodes/1/properties"),
                concat!(
                    "drm_render_minor 128\nvendor_id 4098\ndevice_id 30032\n",
                    "gfx_target_version 120001\ncapability 281019010\n",
                    "simd_count 64\nsimd_per_cu 2\nmax_waves_per_simd 16\n",
                    "max_slots_scratch_cu 32\nwave_front_size 32\n",
                    "lds_size_in_kb 64\narray_count 4\nsimd_arrays_per_engine 2\n",
                    "num_xcc 1\nnum_cp_queues 4\nfw_version 3390\nsdma_fw_version 36502\n",
                    "cwsr_size 65536\nctl_stack_size 4096\nnum_sdma_engines 2\n",
                    "num_sdma_xgmi_engines 1\nnum_gws 32\nfamily_id 145\n",
                    "max_engine_clk_fcompute 2900\n",
                    "num_sdma_queues_per_engine 6\nlocation_id 256\ndomain 0\nunique_id 123\n",
                    "hive_id 99\ncaches_count 0\nio_links_count 0\np2p_links_count 0\n"
                ),
            )
            .unwrap();
            for (name, value) in [
                ("mem_info_vram_total", "8589934592"),
                ("mem_info_vis_vram_total", "268435456"),
                ("vendor", "0x1002"),
                ("device", "0x7550"),
                ("subsystem_vendor", "0x1002"),
                ("subsystem_device", "0x0e3a"),
                ("revision", "0xc0"),
            ] {
                std::fs::write(root.join("drm/renderD128/device").join(name), value).unwrap();
            }
            Self(root)
        }
        fn roots(&self) -> (String, String) {
            (
                self.0.join("topology").to_str().unwrap().into(),
                self.0.join("drm").to_str().unwrap().into(),
            )
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn memory_bus_facts_come_from_nonempty_vram_banks() {
        let fixture = Fixture::new();
        let properties = fixture.0.join("topology/nodes/1/properties");
        let mut contents = std::fs::read_to_string(&properties).unwrap();
        contents.push_str("mem_banks_count 3\n");
        std::fs::write(&properties, contents).unwrap();
        for (ordinal, heap, size, width, clock) in [
            (0, 0, 16_384, 128, 2000),
            (1, 2, 1_048_576, 384, 1400),
            (2, 1, 0, 32, 10),
        ] {
            let directory = fixture
                .0
                .join(format!("topology/nodes/1/mem_banks/{ordinal}"));
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join("properties"),
                format!(
                    "heap_type {heap}\nsize_in_bytes {size}\nwidth {width}\nmem_clk_max {clock}\n"
                ),
            )
            .unwrap();
        }
        let (root, drm) = fixture.roots();
        let endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
        )
        .unwrap();
        let gpu = endpoint.gpu().unwrap();
        assert_eq!(gpu.memory_bus_width_bits, 384);
        assert_eq!(gpu.maximum_memory_clock_mhz, 1400);
        let snapshot = topology(&root, &drm);
        let gpu_node = snapshot
            .nodes
            .iter()
            .find(|node| node.native_id == NativeNodeId::LinuxKfd(1))
            .unwrap();
        let banks = snapshot.banks_for_node(gpu_node.id).collect::<Vec<_>>();
        assert_eq!(banks.len(), 3);
        assert_eq!(banks[1].native_ordinal, 1);
        assert_eq!(banks[1].kind, MemoryBankKind::FrameBufferPrivate);
        assert_eq!(banks[1].capacity_bytes, 1_048_576);
        assert_eq!(banks[1].width_bits, Some(384));
        assert_eq!(banks[1].maximum_clock_mhz, Some(1400));
    }

    #[test]
    fn passive_records_are_complete_and_direct_open_ignores_other_nodes() {
        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let mut found = None;
        enumerate(&root, &drm, &mut |e| {
            assert!(found.is_none());
            found = Some(e);
            Ok(())
        })
        .unwrap();
        let endpoint = found.unwrap();
        assert_eq!(endpoint.driver_instance, DriverInstanceId::TEST);
        let linux = crate::topology::platform::linux::kfd_drm_info(&endpoint).unwrap();
        assert_eq!(linux.node_id, 1);
        assert_eq!(linux.gpu_id, 42);
        let gpu = endpoint.gpu().unwrap();
        assert_eq!(gpu.unique_id, Some(123));
        assert_eq!(gpu.hive_id, 99);
        assert_eq!(gpu.asic_family_id, 145);
        assert_eq!(gpu.maximum_engine_clock_mhz, 2900);
        assert_eq!(gpu.compute_unit_count, 32);
        assert_eq!(gpu.simd_count_per_compute_unit, 2);
        assert_eq!(gpu.shader_engine_count_per_xcc, 2);
        assert_eq!(gpu.shader_array_count_per_engine, 2);
        assert_eq!(gpu.sdma_xgmi_engine_count, 1);
        assert_eq!(gpu.gws_count, 32);
        assert!(gpu.iommu_v2_supported);
        assert!(gpu.coherent_host_access);
        assert_eq!(gpu.maximum_wave_count_per_compute_unit, 32);
        assert_eq!(gpu.maximum_address_watch_point_count, 4);
        assert_eq!(gpu.asic_revision, 3);
        assert_eq!(gpu.packet_processor_firmware_version, 318);
        assert_eq!(gpu.sdma_firmware_version, 662);
        assert_eq!(gpu.local_data_share_byte_length, 65536);
        assert_eq!(endpoint.address_bit_count, 47);
        assert_eq!(endpoint.maximum_address, (1u64 << 47) - 1);
        assert_eq!(
            endpoint.supported_permissions,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE
        );
        assert_eq!(
            endpoint.host_local_cacheability,
            Some(HostCacheability::WriteCombined)
        );
        assert!(gpu.queues.aql && gpu.queues.sdma);
        assert_eq!(gpu.queues.pm4, cfg!(target_arch = "x86_64"));
        assert_eq!(
            gpu.queues.aql_system_cache_control,
            cfg!(target_arch = "x86_64")
        );
        std::fs::create_dir(fixture.0.join("topology/nodes/2")).unwrap();
        std::fs::write(fixture.0.join("topology/nodes/2/gpu_id"), "not a number").unwrap();
        let opened = open_endpoint(&root, &drm, endpoint.id).unwrap();
        assert_eq!(opened.id, endpoint.id);
        assert_eq!(opened.selector, endpoint.selector);
        assert!(enumerate(&root, &drm, &mut |_| Ok(())).is_err());
        let mut stale = endpoint.id;
        stale[8] ^= 1;
        assert_eq!(
            open_endpoint(&root, &drm, stale).unwrap_err().kind(),
            ErrorKind::DeviceLost
        );
    }

    #[test]
    fn probe_distinguishes_unknown_ids_from_incomplete_known_nodes() {
        const ID: [u8; 16] = [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0];
        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let mut unknown = ID;
        unknown[0] = 2;
        assert!(probe_endpoint(&root, &drm, unknown).unwrap().is_none());
        let mut stale = ID;
        stale[8] += 1;
        assert_eq!(
            probe_endpoint(&root, &drm, stale).unwrap_err().kind(),
            ErrorKind::DeviceLost
        );
        assert_eq!(probe_endpoint(&root, &drm, ID).unwrap().unwrap().id, ID);

        let properties = fixture.0.join("topology/nodes/1/properties");
        let contents = std::fs::read_to_string(&properties).unwrap();
        std::fs::write(
            &properties,
            contents.replace("caches_count 0", "caches_count 1"),
        )
        .unwrap();
        assert!(probe_endpoint(&root, &drm, ID).is_err());

        std::fs::write(
            &properties,
            contents.replace("io_links_count 0", "io_links_count 1"),
        )
        .unwrap();
        let link = fixture.0.join("topology/nodes/1/io_links/0");
        std::fs::create_dir_all(&link).unwrap();
        std::fs::write(
            link.join("properties"),
            "type 2\nnode_from 1\nnode_to 2\nflags 1\n",
        )
        .unwrap();
        assert_eq!(
            probe_endpoint(&root, &drm, ID)
                .unwrap_err()
                .native_error_code(),
            Some(errno::ENOENT)
        );
    }

    #[test]
    fn topology_link_access_follows_rocr_types_flags_and_distance() -> Result<(), Error> {
        let fixture = Fixture::new();
        let properties = fixture.0.join("topology/nodes/1/properties");
        let contents = std::fs::read_to_string(&properties).unwrap();
        std::fs::write(
            &properties,
            contents
                .replace("io_links_count 0", "io_links_count 1")
                .replace("p2p_links_count 0", "p2p_links_count 4"),
        )
        .unwrap();
        for (node, gpu_id) in [(2, 53), (3, 64), (4, 75), (5, 86), (6, 97)] {
            std::fs::create_dir_all(fixture.0.join(format!("topology/nodes/{node}"))).unwrap();
            std::fs::write(
                fixture.0.join(format!("topology/nodes/{node}/gpu_id")),
                gpu_id.to_string(),
            )
            .unwrap();
        }
        for (directory, ordinal, kind, target, flags) in [
            ("io_links", 0, 11, 2, 0),
            ("p2p_links", 0, 2, 3, 1),
            ("p2p_links", 1, 2, 4, 1 | (1 << 4)),
            ("p2p_links", 2, 2, 5, 1 | (1 << 5)),
            ("p2p_links", 3, 11, 6, 1),
        ] {
            let path = fixture
                .0
                .join(format!("topology/nodes/1/{directory}/{ordinal}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join("properties"),
                format!("type {kind}\nnode_from 1\nnode_to {target}\nweight 20\nflags {flags}\n"),
            )
            .unwrap();
        }
        let (root, drm) = fixture.roots();
        let endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
        )
        .unwrap();
        let mut owner = endpoint.clone();
        assert!(super::can_access_local_memory(&endpoint, &owner));
        owner.driver_instance = crate::driver::new_driver_instance();
        assert!(!super::can_access_local_memory(&endpoint, &owner));
        owner.driver_instance = endpoint.driver_instance;
        let crate::driver::EndpointSelector::LinuxKfd {
            mut native,
            access_targets,
        } = owner.selector.clone()
        else {
            return Err(error(ErrorKind::InvalidData, "test endpoint is not KFD"));
        };
        native.gpu_id += 1;
        owner.selector = crate::driver::EndpointSelector::LinuxKfd {
            native,
            access_targets: access_targets.clone(),
        };
        assert!(!super::can_access_local_memory(&endpoint, &owner));
        for (node, gpu_id, expected) in [
            (2, 53, true),
            (3, 64, true),
            (4, 75, false),
            (5, 86, true),
            (6, 97, true),
        ] {
            native.node = node;
            native.gpu_id = gpu_id;
            owner.selector = crate::driver::EndpointSelector::LinuxKfd {
                native,
                access_targets: access_targets.clone(),
            };
            assert_eq!(super::can_access_local_memory(&endpoint, &owner), expected);
        }
        Ok(())
    }

    #[test]
    fn endpoint_from_one_driver_cannot_activate_in_another_session() {
        use crate::driver::{Driver, DriverInstance};

        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let mut endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
        )
        .unwrap();
        let first =
            crate::session::Session::new(crate::session::DriverContextLifetime::Process).unwrap();
        let second =
            crate::session::Session::new(crate::session::DriverContextLifetime::Process).unwrap();
        endpoint.driver_instance = first
            .installed_drivers()
            .unwrap()
            .iter()
            .find_map(DriverInstance::linux_kfd)
            .unwrap()
            .driver_instance();
        assert_eq!(
            second.activate(&endpoint).err().unwrap().kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn cpu_gpu_links_preserve_independent_directional_metadata() {
        let fixture = Fixture::new();
        let gpu_properties = fixture.0.join("topology/nodes/1/properties");
        let contents = std::fs::read_to_string(&gpu_properties).unwrap();
        std::fs::write(
            &gpu_properties,
            contents.replace("io_links_count 0", "io_links_count 1"),
        )
        .unwrap();
        std::fs::create_dir_all(fixture.0.join("topology/nodes/0/io_links/0")).unwrap();
        std::fs::write(fixture.0.join("topology/nodes/0/gpu_id"), "0\n").unwrap();
        std::fs::write(
            fixture.0.join("topology/nodes/0/properties"),
            "cpu_cores_count 8\ncpu_core_id_base 16\nmax_engine_clk_ccompute 4200\n\
             mem_banks_count 1\n\
             io_links_count 1\np2p_links_count 0\n",
        )
        .unwrap();
        let bank = fixture.0.join("topology/nodes/0/mem_banks/0");
        std::fs::create_dir_all(&bank).unwrap();
        std::fs::write(
            bank.join("properties"),
            "heap_type 0\nsize_in_bytes 1073741824\nflags 3\n",
        )
        .unwrap();
        std::fs::create_dir_all(fixture.0.join("topology/nodes/1/io_links/0")).unwrap();
        std::fs::write(
            fixture.0.join("topology/nodes/1/io_links/0/properties"),
            concat!(
                "type 2\nnode_from 1\nnode_to 0\nweight 20\n",
                "min_latency 1\nmax_latency 2\nmin_bandwidth 3\nmax_bandwidth 64000\n",
                "flags 1\n"
            ),
        )
        .unwrap();
        std::fs::write(
            fixture.0.join("topology/nodes/0/io_links/0/properties"),
            concat!(
                "type 2\nnode_from 0\nnode_to 1\nweight 21\n",
                "min_latency 4\nmax_latency 5\nmin_bandwidth 6\nmax_bandwidth 63000\n",
                "flags 3\n"
            ),
        )
        .unwrap();

        let (root, drm) = fixture.roots();
        let snapshot = topology(&root, &drm);
        let endpoint = &snapshot.endpoints[0];
        assert_eq!(snapshot.sources.last().unwrap().generation, Some(1));
        assert!(snapshot.nodes.iter().any(|node| {
            node.native_id == NativeNodeId::LinuxKfd(0)
                && node.kind == NodeKind::KfdHost
                && node.cpu
                    == Some(CpuNodeInfo {
                        core_count: 8,
                        core_id_base: Some(16),
                        maximum_clock_mhz: Some(4200),
                    })
        }));
        let cpu_node = snapshot
            .nodes
            .iter()
            .find(|node| node.native_id == NativeNodeId::LinuxKfd(0))
            .unwrap();
        let bank = snapshot.banks_for_node(cpu_node.id).next().unwrap();
        assert_eq!(bank.kind, MemoryBankKind::System);
        assert_eq!(bank.capacity_bytes, 1_073_741_824);
        assert_eq!(bank.native_flags, 3);
        let to_host = snapshot.route_to_host(endpoint).unwrap();
        assert_eq!(to_host.kind, MemoryLinkType::Pcie);
        assert_eq!(to_host.minimum_latency, 1);
        assert_eq!(to_host.maximum_latency, 2);
        assert_eq!(to_host.minimum_bandwidth, 3);
        assert_eq!(to_host.maximum_bandwidth, 64_000);
        assert!(to_host.atomic_support_32bit);
        assert!(to_host.atomic_support_64bit);
        assert!(to_host.coherent_support);
        assert_eq!(to_host.numa_distance, 20);
        assert_eq!(to_host.hop_count(), 1);

        let from_host = snapshot.route_from_host(endpoint).unwrap();
        assert_eq!(from_host.kind, MemoryLinkType::Pcie);
        assert_eq!(from_host.minimum_latency, 4);
        assert_eq!(from_host.maximum_latency, 5);
        assert_eq!(from_host.minimum_bandwidth, 6);
        assert_eq!(from_host.maximum_bandwidth, 63_000);
        assert!(from_host.atomic_support_32bit);
        assert!(from_host.atomic_support_64bit);
        assert!(!from_host.coherent_support);
        assert_eq!(from_host.numa_distance, 21);
        assert_eq!(from_host.hop_count(), 1);
    }

    #[test]
    fn host_facts_include_caches_and_one_way_native_routes() {
        let fixture = Fixture::new();
        for node in [0, 2] {
            let path = fixture.0.join(format!("topology/nodes/{node}"));
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(path.join("gpu_id"), "0\n").unwrap();
            std::fs::write(
                path.join("properties"),
                if node == 0 {
                    "caches_count 1\nio_links_count 2\np2p_links_count 0\n"
                } else {
                    "caches_count 0\nio_links_count 0\np2p_links_count 0\n"
                },
            )
            .unwrap();
        }
        let cache = fixture.0.join("topology/nodes/0/caches/0");
        std::fs::create_dir_all(&cache).unwrap();
        std::fs::write(
            cache.join("properties"),
            concat!(
                "processor_id_low 0\nlevel 3\nsize 512\ncache_line_size 64\n",
                "cache_lines_per_tag 1\nassociation 16\nlatency 1\n",
                "type 5\nsibling_map 1,0,0,0\n"
            ),
        )
        .unwrap();
        for (ordinal, target) in [(0, 1), (1, 2)] {
            let link = fixture
                .0
                .join(format!("topology/nodes/0/io_links/{ordinal}"));
            std::fs::create_dir_all(&link).unwrap();
            std::fs::write(
                link.join("properties"),
                format!("type 2\nnode_from 0\nnode_to {target}\nweight 20\nflags 1\n"),
            )
            .unwrap();
        }
        let (root, drm) = fixture.roots();
        let snapshot = topology(&root, &drm);
        let host = snapshot
            .nodes
            .iter()
            .find(|node| node.native_id == NativeNodeId::LinuxKfd(0))
            .unwrap();
        let peer_host = snapshot
            .nodes
            .iter()
            .find(|node| node.native_id == NativeNodeId::LinuxKfd(2))
            .unwrap();
        assert!(snapshot.caches.iter().any(|cache| {
            cache.node == host.id
                && cache.info.cpu_cache
                && cache.info.level == 3
                && cache.info.size_bytes == 512 * 1024
        }));
        assert!(
            snapshot
                .routes
                .iter()
                .any(|route| route.source == host.id && route.target == peer_host.id)
        );
        assert!(snapshot.route_from_host(&snapshot.endpoints[0]).is_some());
        assert!(snapshot.route_to_host(&snapshot.endpoints[0]).is_none());
        assert_eq!(snapshot.sources.last().unwrap().generation, Some(1));
    }

    #[test]
    fn restricted_peer_route_is_reported_without_granting_local_access() {
        let fixture = Fixture::new();
        let first_properties = fixture.0.join("topology/nodes/1/properties");
        let properties = std::fs::read_to_string(&first_properties).unwrap();
        std::fs::write(
            &first_properties,
            properties.replace("p2p_links_count 0", "p2p_links_count 1"),
        )
        .unwrap();
        let second = fixture.0.join("topology/nodes/2");
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(second.join("gpu_id"), "43\n").unwrap();
        std::fs::write(second.join("name"), "second GPU\n").unwrap();
        std::fs::write(
            second.join("properties"),
            properties.replace("unique_id 123", "unique_id 124"),
        )
        .unwrap();
        let link = fixture.0.join("topology/nodes/1/p2p_links/0");
        std::fs::create_dir_all(&link).unwrap();
        std::fs::write(
            link.join("properties"),
            "type 2\nnode_from 1\nnode_to 2\nweight 20\nflags 17\n",
        )
        .unwrap();
        let (root, drm) = fixture.roots();
        let snapshot = topology(&root, &drm);
        let first = snapshot
            .endpoints
            .iter()
            .find(|endpoint| endpoint.id[0] == 1)
            .unwrap();
        let second = snapshot
            .endpoints
            .iter()
            .find(|endpoint| endpoint.id[0] == 2)
            .unwrap();
        assert_eq!(
            snapshot.route_between(first, second).unwrap().kind,
            MemoryLinkType::Pcie
        );
        assert!(!super::can_access_local_memory(first, second));
    }

    #[test]
    fn cache_records_convert_kib_and_preserve_order_and_types() {
        let fixture = Fixture::new();
        let properties = fixture.0.join("topology/nodes/1/properties");
        let contents = std::fs::read_to_string(&properties).unwrap();
        std::fs::write(
            &properties,
            contents.replace("caches_count 0", "caches_count 4"),
        )
        .unwrap();
        for (ordinal, level, size, kind) in [
            (3, 2, 256, 8),
            (1, 1, 32, 10),
            (0, 1, 32, 9),
            (2, 3, 8192, 5),
        ] {
            let directory = fixture.0.join(format!("topology/nodes/1/caches/{ordinal}"));
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join("properties"),
                format!(
                    "processor_id_low 0\nlevel {level}\nsize {size}\ncache_line_size 64\ncache_lines_per_tag 1\nassociation 16\nlatency 1\ntype {kind}\nsibling_map 1,0,0,0\n"
                ),
            )
            .unwrap();
        }
        let (root, drm) = fixture.roots();
        let snapshot = topology(&root, &drm);
        let endpoint = &snapshot.endpoints[0];
        assert!(snapshot.memory_locations.iter().any(|location| {
            location.kind == MemoryLocationKind::DeviceLocal
                && location.capacity_bytes == Some(endpoint.local_memory_bytes)
                && location.host_visible_bytes == Some(endpoint.host_visible_local_memory_bytes)
        }));
        let caches = snapshot
            .caches_for_endpoint(endpoint)
            .copied()
            .collect::<Vec<_>>();
        assert_eq!(caches.len(), 4);
        assert_eq!(caches[0].level, 1);
        assert_eq!(caches[0].size_bytes, 32_768);
        assert!(crate::device::gpu::is_compute_data_cache(&caches[0]));
        assert!(!crate::device::gpu::is_compute_data_cache(&caches[1]));
        assert!(!crate::device::gpu::is_compute_data_cache(&caches[2]));
        assert!(crate::device::gpu::is_compute_data_cache(&caches[3]));
        assert_eq!(caches[3].level, 2);
        assert_eq!(caches[3].size_bytes, 262_144);
        assert_eq!(caches[3].line_size_bytes, Some(64));
        assert_eq!(endpoint.gpu().unwrap().persisting_l2_cache_size_max, 0);
    }

    #[test]
    fn incomplete_or_mismatched_cache_topology_rejects_the_endpoint() {
        for contents in [
            None,
            Some(
                "processor_id_low 0\nlevel 1\nsize 32768\ncache_line_size 64\ncache_lines_per_tag 1\nassociation 16\nlatency 1\ntype 9\n",
            ),
        ] {
            let fixture = Fixture::new();
            let properties = fixture.0.join("topology/nodes/1/properties");
            let node = std::fs::read_to_string(&properties).unwrap();
            std::fs::write(
                &properties,
                node.replace("caches_count 0", "caches_count 1"),
            )
            .unwrap();
            std::fs::create_dir_all(fixture.0.join("topology/nodes/1/caches")).unwrap();
            if let Some(contents) = contents {
                let directory = fixture.0.join("topology/nodes/1/caches/0");
                std::fs::create_dir_all(&directory).unwrap();
                std::fs::write(directory.join("properties"), contents).unwrap();
            }
            let (root, drm) = fixture.roots();
            assert_eq!(
                open_endpoint(
                    &root,
                    &drm,
                    [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
                )
                .unwrap_err()
                .kind(),
                ErrorKind::InvalidData
            );
        }
    }

    #[test]
    fn malformed_link_source_or_count_rejects_the_endpoint_snapshot() {
        for source in [Some(2), None] {
            let fixture = Fixture::new();
            let properties = fixture.0.join("topology/nodes/1/properties");
            let contents = std::fs::read_to_string(&properties).unwrap();
            std::fs::write(
                &properties,
                contents.replace("io_links_count 0", "io_links_count 1"),
            )
            .unwrap();
            let directory = fixture.0.join("topology/nodes/1/io_links");
            std::fs::create_dir_all(&directory).unwrap();
            if let Some(source) = source {
                let link = directory.join("0");
                std::fs::create_dir_all(&link).unwrap();
                std::fs::write(
                    link.join("properties"),
                    format!("type 11\nnode_from {source}\nnode_to 2\nflags 1\n"),
                )
                .unwrap();
            }
            let (root, drm) = fixture.roots();
            let error = open_endpoint(
                &root,
                &drm,
                [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidData);
        }
    }

    #[test]
    fn compute_cache_control_requires_the_qualified_target() {
        for gfx in [120_000, 120_001, 120_500, 110_000] {
            let fixture = Fixture::new();
            let path = fixture.0.join("topology/nodes/1/properties");
            let properties = std::fs::read_to_string(&path).unwrap();
            std::fs::write(
                path,
                properties.replace(
                    "gfx_target_version 120001",
                    &format!("gfx_target_version {gfx}"),
                ),
            )
            .unwrap();
            let (root, drm) = fixture.roots();
            let mut found = None;
            enumerate(&root, &drm, &mut |endpoint| {
                found = Some(endpoint);
                Ok(())
            })
            .unwrap();
            let endpoint = found.unwrap();
            let queues = endpoint.gpu().unwrap().queues;
            let supported = matches!(gfx, 120_000 | 120_001 | 120_500);
            assert!(queues.aql);
            assert_eq!(
                queues.aql_system_cache_control,
                cfg!(target_arch = "x86_64") && supported
            );
            assert_eq!(
                queues.pm4_system_cache_control,
                cfg!(target_arch = "x86_64") && supported
            );
            assert_eq!(queues.pm4, cfg!(target_arch = "x86_64") && supported);
        }
    }

    #[test]
    fn local_cacheability_requires_a_qualified_target_and_visible_aperture() {
        for (gfx, device, visible, expected) in [
            (90000, 0x7550, 268_435_456, None),
            (90010, 0x7550, 268_435_456, None),
            (90402, 0x74a1, 268_435_456, None),
            (90500, 0x7550, 268_435_456, None),
            (
                100_100,
                0x7550,
                268_435_456,
                Some(HostCacheability::WriteCombined),
            ),
            (
                120_000,
                0x7550,
                268_435_456,
                Some(HostCacheability::WriteCombined),
            ),
            (
                120_001,
                0x7550,
                268_435_456,
                Some(HostCacheability::WriteCombined),
            ),
            (120_001, 0x7550, 0, None),
            (
                120_500,
                0x7550,
                268_435_456,
                Some(HostCacheability::WriteCombined),
            ),
        ] {
            let fixture = Fixture::new();
            let path = fixture.0.join("topology/nodes/1/properties");
            let properties = std::fs::read_to_string(&path).unwrap();
            std::fs::write(
                path,
                properties
                    .replace(
                        "gfx_target_version 120001",
                        &format!("gfx_target_version {gfx}"),
                    )
                    .replace("device_id 30032", &format!("device_id {device}")),
            )
            .unwrap();
            std::fs::write(
                fixture.0.join("drm/renderD128/device/device"),
                device.to_string(),
            )
            .unwrap();
            std::fs::write(
                fixture
                    .0
                    .join("drm/renderD128/device/mem_info_vis_vram_total"),
                visible.to_string(),
            )
            .unwrap();
            let (root, drm) = fixture.roots();
            let mut found = None;
            enumerate(&root, &drm, &mut |e| {
                found = Some(e);
                Ok(())
            })
            .unwrap();
            let endpoint = found.unwrap();
            assert_eq!(endpoint.host_local_cacheability, expected);
            assert_eq!(endpoint.local_memory_bytes, 8_589_934_592);
            assert_eq!(endpoint.host_visible_local_memory_bytes, visible);
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn passive_enumeration_fails_before_visiting_on_metadata_exhaustion() {
        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let callbacks = crate::test_support::allocator::State::default();
        // SAFETY: Test state outlives the scoped metadata fault hook.
        let _scope = unsafe { callbacks.allocator() }.scope();
        let mut endpoint = None;
        enumerate(&root, &drm, &mut |e| {
            endpoint = Some(e);
            Ok(())
        })
        .unwrap();
        assert!(callbacks.allocations.load(Ordering::Relaxed) >= 1);
        assert_eq!(
            callbacks.allocations.load(Ordering::Relaxed),
            callbacks.frees.load(Ordering::Relaxed)
        );
        let session =
            crate::session::Session::new(crate::session::DriverContextLifetime::Session).unwrap();
        assert_eq!(
            session.activate(&endpoint.unwrap()).err().unwrap().kind(),
            ErrorKind::InvalidArgument
        );
        callbacks.fail.store(true, Ordering::Relaxed);
        let mut visited = false;
        assert_eq!(
            enumerate(&root, &drm, &mut |_| {
                visited = true;
                Ok(())
            })
            .unwrap_err()
            .kind(),
            ErrorKind::ResourceExhausted
        );
        assert!(!visited);
    }

    #[test]
    fn generated_topology_and_cache_mutations_stay_bounded() {
        const NODE: &str = "simd_count 64\nmax_slots_scratch_cu 32\nnum_xcc 1\n";
        const CACHE: &str = concat!(
            "processor_id_low 0\nlevel 2\nsize 512\ncache_line_size 64\n",
            "cache_lines_per_tag 1\nassociation 16\nlatency 1\n",
            "type 2\nsibling_map 1,0,0,0\n"
        );
        assert!(Properties::new(NODE).is_ok());
        assert!(cache_properties(CACHE).is_ok());
        let mut seed = 0xd1b5_4a32_d192_ed03_u64;
        for case in 0..2048 {
            let mut bytes = if case % 2 == 0 {
                NODE.as_bytes().to_vec()
            } else {
                CACHE.as_bytes().to_vec()
            };
            for _ in 0..=(case % 5) {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let index = usize::try_from(seed % u64::try_from(bytes.len()).unwrap()).unwrap();
                bytes[index] = b"0x 19\n,+-_z"[(seed >> 32) as usize % 11];
            }
            let length = if case % 4 == 0 {
                usize::try_from(seed % u64::try_from(bytes.len() + 1).unwrap()).unwrap()
            } else {
                bytes.len()
            };
            let input = String::from_utf8_lossy(&bytes[..length]);
            if let Ok(properties) = Properties::new(&input) {
                let _ = properties.required("simd_count");
                let _ = properties.optional_u32("num_xcc");
            }
            let _ = cache_properties(&input);
        }
    }

    #[test]
    fn parser_rejects_incomplete_and_duplicate_geometry() {
        for value in [
            "simd_count 1\nsimd_count 2\n",
            "simd_count\n",
            "simd_count 1 extra\n",
            "simd_count -1\n",
        ] {
            assert!(Properties::new(value).is_err());
        }
        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let path = fixture.0.join("topology/nodes/1/properties");
        let properties = std::fs::read_to_string(&path).unwrap();
        std::fs::write(path, properties.replace("max_slots_scratch_cu 32\n", "")).unwrap();
        let mut visited = false;
        assert_eq!(
            enumerate(&root, &drm, &mut |_| {
                visited = true;
                Ok(())
            })
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidData
        );
        assert!(!visited);
    }

    #[test]
    fn persisting_l2_limit_comes_from_the_gpu_node() {
        let fixture = Fixture::new();
        let properties = fixture.0.join("topology/nodes/1/properties");
        let mut contents = std::fs::read_to_string(&properties).unwrap();
        contents = contents.replace("caches_count 0", "caches_count 3");
        contents.push_str("persisting_cache_size_max 2097152\n");
        std::fs::write(&properties, contents).unwrap();
        for (ordinal, level, max) in [(0, 1, 4_194_304), (1, 2, 0), (2, 2, 1_048_576)] {
            let directory = fixture.0.join(format!("topology/nodes/1/caches/{ordinal}"));
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(
                directory.join("properties"),
                format!(
                    "processor_id_low 0\nlevel {level}\nsize 256\ncache_line_size 64\ncache_lines_per_tag 1\nassociation 16\nlatency 1\ntype 8\npersisting_cache_size_max {max}\nsibling_map 1,0,0,0\n"
                ),
            )
            .unwrap();
        }
        let (root, drm) = fixture.roots();
        let endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
        )
        .unwrap();
        assert_eq!(
            endpoint.gpu().unwrap().persisting_l2_cache_size_max,
            1_048_576
        );
    }
}
