//! Passive Linux topology discovery and validation.
//!
//! Discovery reads fixed-size sysfs records and opens no native execution
//! endpoint. Endpoint IDs select one node directly and are checked against its
//! current native identity before publication and again before activation.
//! Parsers reject incomplete, duplicate, overflowing, or internally inconsistent
//! properties rather than filling important device facts with defaults.
use super::memory::{error, native_error};
use super::sys;
use crate::host_storage::{Allocator, Buffer, Shared};
use crate::memory::{DeviceAccess, HostCacheability};
use crate::topology::{
    CacheInfo, Endpoint, EndpointKind, GpuInfo, GpuQueueCapabilities, MemoryLink, MemoryLinkInfo,
    MemoryLinkType, PciInfo, TopologyKey,
};
use crate::{Error, ErrorKind};
use std::fmt::{self, Write as _};
use std::fs::{File, OpenOptions};
use std::io::{self, Read as _};
use std::path::Path;

/// Stable native identity and capabilities captured for one topology node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeNode {
    pub node: u32,
    pub gpu_id: u32,
    pub render_minor: Option<u32>,
    pub unique_id: Option<u64>,
    pub identity: [u8; 16],
    pub queues: NativeQueueProperties,
    pub local_memory_bytes: u64,
    pub public_memory_bytes: u64,
}

/// Queue and shader properties required to qualify native transports.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct NativeQueueProperties {
    pub gfx_target: u32,
    pub compute_units: u32,
    pub maximum_wave_count_per_compute_unit: u32,
    pub maximum_scratch_wave_count_per_compute_unit: u32,
    pub wavefront_size: u32,
    pub xcc_count: u32,
    pub shader_engine_count_per_xcc: u32,
    pub context_size: u32,
    pub control_stack_size: u32,
    pub sdma_engines: u32,
    pub compute_queues: u32,
    pub sdma_qualified: bool,
}

struct StackPath {
    bytes: [u8; 512],
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

fn cache_properties(text: &str) -> Result<CacheInfo, Error> {
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
    let _ = properties.u32("cache_line_size")?;
    let _ = properties.u32("cache_lines_per_tag")?;
    let _ = properties.u32("association")?;
    let _ = properties.u32("latency")?;
    Ok(CacheInfo {
        level: properties.u32("level")?,
        size: properties.u32("size")?,
        kind: properties.u32("type")?,
    })
}

fn caches(
    root: &str,
    node: u32,
    count: u32,
    allocator: Allocator,
) -> Result<Option<Shared<Buffer<CacheInfo>>>, Error> {
    if count == 0 {
        return Ok(None);
    }
    let capacity = usize::try_from(count)
        .map_err(|_| error(ErrorKind::ResourceExhausted, "native cache count overflow"))?;
    let mut records = Buffer::try_with_capacity(capacity, allocator)?;
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
    let mut cache_info = Buffer::try_with_capacity(capacity, allocator)?;
    for (_, cache) in records {
        cache_info.try_push(cache)?;
    }
    Shared::new(cache_info, allocator)
        .map(Some)
        .map_err(Into::into)
}

pub(super) fn enumerate(
    root: &str,
    drm: &str,
    allocator: Allocator,
    visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
) -> Result<(), Error> {
    for _ in 0..3 {
        let before = match scalar(format_args!("{root}/generation_id")) {
            Ok(value) => value,
            Err(e) if e.native_error_code() == Some(2) => return Ok(()),
            Err(e) => return Err(e),
        };
        let mut records = Buffer::new(allocator);
        let nodes = StackPath::new(format_args!("{root}/nodes"))?;
        let result = sys::numeric_directories(nodes.path(), &mut |node| {
            if let Some(endpoint) = read_node(root, drm, node, allocator)? {
                records.try_push(endpoint)?;
            }
            Ok(())
        });
        let after = scalar(format_args!("{root}/generation_id"))?;
        if before != after {
            continue;
        }
        result?;
        records.sort_unstable_by_key(|record| record.native.node);
        for endpoint in records {
            visitor(endpoint)?;
        }
        return Ok(());
    }
    Err(error(
        ErrorKind::ConcurrentModification,
        "native topology changed during every read",
    ))
}

pub(super) fn open_endpoint(
    root: &str,
    drm: &str,
    id: [u8; 16],
    allocator: Allocator,
) -> Result<Endpoint, Error> {
    let node = u32::from_le_bytes([id[0], id[1], id[2], id[3]]);
    let before = scalar(format_args!("{root}/generation_id"))?;
    let endpoint = read_node(root, drm, node, allocator)?
        .ok_or_else(|| error(ErrorKind::DeviceLost, "native GPU endpoint disappeared"))?;
    let after = scalar(format_args!("{root}/generation_id"))?;
    if before != after {
        return Err(error(
            ErrorKind::ConcurrentModification,
            "native endpoint changed during read",
        ));
    }
    if endpoint.id != id {
        return Err(error(
            ErrorKind::DeviceLost,
            "native endpoint identity changed",
        ));
    }
    Ok(endpoint)
}

fn kfd_link(
    node: u32,
    gpu_id: u32,
    target: u32,
    target_gpu_id: u32,
    info: MemoryLinkInfo,
) -> MemoryLink {
    MemoryLink {
        source: TopologyKey {
            group: node,
            member: gpu_id,
        },
        target: TopologyKey {
            group: target,
            member: target_gpu_id,
        },
        source_is_host: gpu_id == 0,
        target_is_host: target_gpu_id == 0,
        info,
    }
}

fn node_memory_links(
    root: &str,
    node: u32,
    gpu_id: u32,
    io_link_count: u32,
    p2p_link_count: u32,
    target_filter: Option<u32>,
    allocator: Allocator,
) -> Result<Buffer<MemoryLink>, Error> {
    const OVERRIDE: u32 = 1;
    const NON_COHERENT: u32 = 1 << 1;
    const NO_ATOMICS_32: u32 = 1 << 2;
    const NO_ATOMICS_64: u32 = 1 << 3;
    const NO_PEER_TO_PEER_DMA: u32 = 1 << 4;
    let capacity = usize::try_from(io_link_count)
        .ok()
        .and_then(|count| count.checked_add(p2p_link_count as usize))
        .ok_or_else(|| error(ErrorKind::ResourceExhausted, "native link count overflow"))?;
    let mut links: Buffer<MemoryLink> = Buffer::try_with_capacity(capacity, allocator)?;
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
            if target_filter.is_some_and(|target| target != to) {
                return Ok(());
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
            if flags & OVERRIDE != 0 && flags & NO_PEER_TO_PEER_DMA != 0 {
                return Ok(());
            }
            let overridden = flags & OVERRIDE != 0;
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
            let link = kfd_link(node, gpu_id, to, target_gpu_id, info);
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
    allocator: Allocator,
) -> Result<Option<Shared<Buffer<MemoryLink>>>, Error> {
    let mut links = node_memory_links(
        root,
        node,
        gpu_id,
        io_link_count,
        p2p_link_count,
        None,
        allocator,
    )?;
    let outgoing_count = links.len();
    let mut host_nodes = Buffer::try_with_capacity(outgoing_count, allocator)?;
    for link in links.iter().take(outgoing_count) {
        if link.target_is_host && !host_nodes.iter().any(|node| *node == link.target.group) {
            host_nodes.try_push(link.target.group)?;
        }
    }
    for host_node in host_nodes {
        let path = StackPath::new(format_args!("{root}/nodes/{host_node}/properties"))?;
        let mut bytes = [0; 16384];
        let properties = Properties::new(read(path.path(), &mut bytes)?)?;
        let incoming = node_memory_links(
            root,
            host_node,
            0,
            properties.optional_u32("io_links_count")?.unwrap_or(0),
            properties.optional_u32("p2p_links_count")?.unwrap_or(0),
            Some(node),
            allocator,
        )?;
        for link in incoming {
            if link.target.member != gpu_id {
                return Err(error(
                    ErrorKind::InvalidData,
                    "native reverse link target does not match its GPU",
                ));
            }
            if let Some(existing) = links
                .iter_mut()
                .find(|existing| existing.source == link.source && existing.target == link.target)
            {
                *existing = link;
            } else {
                links.try_push(link)?;
            }
        }
    }
    if links.is_empty() {
        Ok(None)
    } else {
        Shared::new(links, allocator).map(Some).map_err(Into::into)
    }
}

#[allow(
    clippy::too_many_lines,
    reason = "all immutable native facts share one identity and generation bracket"
)]
fn read_node(
    root: &str,
    drm: &str,
    node: u32,
    allocator: Allocator,
) -> Result<Option<Endpoint>, Error> {
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
    let sdma_xgmi_engines = p.optional_u32("num_sdma_xgmi_engines")?.unwrap_or(0);
    let gws_count = p.optional_u32("num_gws")?.unwrap_or(0);
    let io_link_count = p.optional_u32("io_links_count")?.unwrap_or(0);
    let p2p_link_count = p.optional_u32("p2p_links_count")?.unwrap_or(0);
    let caches = caches(
        root,
        node,
        p.optional_u32("caches_count")?.unwrap_or(0),
        allocator,
    )?;
    let memory_links = memory_links(root, node, gpu_id, io_link_count, p2p_link_count, allocator)?;
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
            "incomplete native context metadata",
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
    // doorbells across the already supported GFX10.1–GFX12.0 native targets.
    // Commands remain target-specific and caller-owned; a discovered SDMA IP
    // does not change any frontend's advertised transport revision.
    let sdma_qualified = (100_100..=120_001).contains(&gfx)
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
    let queues = NativeQueueProperties {
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
    let native = NativeNode {
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
    Ok(Some(Endpoint {
        id,
        name,
        pci: Some(pci),
        kind: EndpointKind::Gpu {
            info: GpuInfo {
                gfx_major: gfx / 10000,
                gfx_minor: (gfx / 100) % 100,
                gfx_stepping: gfx % 100,
                unique_id,
                hive_id,
                asic_family_id: family_id,
                asic_revision: (capability & 0x03c0_0000) >> 22,
                maximum_engine_clock_mhz,
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
                packet_processor_firmware_version: p.optional_u32("fw_version")?.unwrap_or(0)
                    & 0x3ff,
                sdma_firmware_version: p.optional_u32("sdma_fw_version")?.unwrap_or(0) & 0x3ff,
                queues: GpuQueueCapabilities {
                    aql: supports_aql,
                    // System-scope AQL packet fences and a kernel-issued system
                    // release are execution-qualified only on the initial
                    // GFX1201/x86-64 target.
                    aql_system_cache_control: cfg!(target_arch = "x86_64")
                        && supports_aql
                        && gfx == 120_001,
                    pm4: supports_pm4,
                    kernel_pm4: supports_pm4,
                    // The initial PM4 profile uses the GFX12 ACQUIRE_MEM GCR
                    // encoding and is execution-qualified on the same exact target
                    // as construction.
                    pm4_system_cache_control: supports_pm4,
                    sdma: sdma_qualified,
                    kernel_sdma: cfg!(target_arch = "x86_64")
                        && sdma_qualified
                        && gfx == 120_001
                        && sdma_ip == Some((7, 0, 1)),
                    // The first execution-qualified path follows ROCr's GFX12
                    // USER_GCR protocol on SDMA 7.0.1. Keep this narrower than
                    // queue construction: neither a nearby GFX version nor missing
                    // IP data proves that protocol.
                    sdma_system_cache_control: cfg!(target_arch = "x86_64")
                        && sdma_qualified
                        && gfx == 120_001
                        && sdma_ip == Some((7, 0, 1)),
                    sdma_engine_count: if sdma_qualified { sdma.unwrap_or(0) } else { 0 },
                },
            },
        },
        local_memory_bytes: local,
        host_visible_local_memory_bytes: public,
        host_local_cacheability: (cfg!(target_arch = "x86_64")
            && public != 0
            && (100_100..=120_001).contains(&gfx))
        .then_some(HostCacheability::WriteCombined),
        allocation_granularity: 4096,
        address_bit_count: bits,
        minimum_address: 65536,
        maximum_address: (1u64 << bits) - 1,
        supported_permissions: DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        caches,
        memory_links,
        topology_key: TopologyKey {
            group: node,
            member: gpu_id,
        },
        provider_instance: 0,
        native,
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
    fn passive_records_are_complete_and_direct_open_ignores_other_nodes() {
        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let mut found = None;
        enumerate(&root, &drm, Allocator::default(), &mut |e| {
            assert!(found.is_none());
            found = Some(e);
            Ok(())
        })
        .unwrap();
        let endpoint = found.unwrap();
        let linux = endpoint.linux_kfd_drm_info();
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
        let opened = open_endpoint(&root, &drm, endpoint.id, Allocator::default()).unwrap();
        assert_eq!(opened.id, endpoint.id);
        assert_eq!(opened.native, endpoint.native);
        assert!(enumerate(&root, &drm, Allocator::default(), &mut |_| Ok(())).is_err());
        let mut stale = endpoint.id;
        stale[8] ^= 1;
        assert_eq!(
            open_endpoint(&root, &drm, stale, Allocator::default())
                .unwrap_err()
                .kind(),
            ErrorKind::DeviceLost
        );
    }

    #[test]
    fn topology_link_access_follows_rocr_types_flags_and_distance() {
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
            Allocator::default(),
        )
        .unwrap();
        let mut owner = endpoint.clone();
        assert!(endpoint.can_access_local_memory(&owner));
        owner.provider_instance = 1;
        assert!(!endpoint.can_access_local_memory(&owner));
        owner.provider_instance = endpoint.provider_instance;
        owner.topology_key.member += 1;
        assert!(!endpoint.can_access_local_memory(&owner));
        owner.topology_key = TopologyKey {
            group: 2,
            member: 53,
        };
        assert!(endpoint.can_access_local_memory(&owner));
        owner.topology_key = TopologyKey {
            group: 3,
            member: 64,
        };
        assert!(endpoint.can_access_local_memory(&owner));
        owner.topology_key = TopologyKey {
            group: 4,
            member: 75,
        };
        assert!(!endpoint.can_access_local_memory(&owner));
        owner.topology_key = TopologyKey {
            group: 5,
            member: 86,
        };
        assert!(endpoint.can_access_local_memory(&owner));
        owner.topology_key = TopologyKey {
            group: 6,
            member: 97,
        };
        assert!(endpoint.can_access_local_memory(&owner));
    }

    #[test]
    fn endpoint_from_one_provider_cannot_activate_in_another_session() {
        use crate::driver::ProviderDriver;

        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let mut endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
            Allocator::default(),
        )
        .unwrap();
        let first = crate::session::Session::new(crate::session::SessionLifetime::Process).unwrap();
        let second =
            crate::session::Session::new(crate::session::SessionLifetime::Process).unwrap();
        endpoint.provider_instance = first.driver().provider_instance();
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
            "io_links_count 1\np2p_links_count 0\n",
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
        let endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
            Allocator::default(),
        )
        .unwrap();
        let to_host = endpoint.memory_link_to_host().unwrap();
        assert_eq!(to_host.kind(), MemoryLinkType::Pcie);
        assert_eq!(to_host.minimum_latency(), 1);
        assert_eq!(to_host.maximum_latency(), 2);
        assert_eq!(to_host.minimum_bandwidth(), 3);
        assert_eq!(to_host.maximum_bandwidth(), 64_000);
        assert!(to_host.supports_32bit_atomics());
        assert!(to_host.supports_64bit_atomics());
        assert!(to_host.is_coherent());
        assert_eq!(to_host.numa_distance(), 20);
        assert_eq!(to_host.hop_count(), 1);

        let from_host = endpoint.memory_link_from_host().unwrap();
        assert_eq!(from_host.kind(), MemoryLinkType::Pcie);
        assert_eq!(from_host.minimum_latency(), 4);
        assert_eq!(from_host.maximum_latency(), 5);
        assert_eq!(from_host.minimum_bandwidth(), 6);
        assert_eq!(from_host.maximum_bandwidth(), 63_000);
        assert!(from_host.supports_32bit_atomics());
        assert!(from_host.supports_64bit_atomics());
        assert!(!from_host.is_coherent());
        assert_eq!(from_host.numa_distance(), 21);
        assert_eq!(from_host.hop_count(), 1);
    }

    #[test]
    fn cache_records_preserve_topology_order_and_types() {
        let fixture = Fixture::new();
        let properties = fixture.0.join("topology/nodes/1/properties");
        let contents = std::fs::read_to_string(&properties).unwrap();
        std::fs::write(
            &properties,
            contents.replace("caches_count 0", "caches_count 4"),
        )
        .unwrap();
        for (ordinal, level, size, kind) in [
            (3, 2, 262_144, 8),
            (1, 1, 32_768, 10),
            (0, 1, 32_768, 9),
            (2, 3, 8_388_608, 5),
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
        let endpoint = open_endpoint(
            &root,
            &drm,
            [1, 0, 0, 0, 42, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0],
            Allocator::default(),
        )
        .unwrap();
        assert_eq!(endpoint.caches().len(), 4);
        assert_eq!(endpoint.caches()[0].level(), 1);
        assert_eq!(endpoint.caches()[0].size(), 32_768);
        assert!(crate::gpu::is_compute_data_cache(&endpoint.caches()[0]));
        assert!(!crate::gpu::is_compute_data_cache(&endpoint.caches()[1]));
        assert!(!crate::gpu::is_compute_data_cache(&endpoint.caches()[2]));
        assert!(crate::gpu::is_compute_data_cache(&endpoint.caches()[3]));
        assert_eq!(endpoint.caches()[3].level(), 2);
        assert_eq!(endpoint.caches()[3].size(), 262_144);
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
                    Allocator::default(),
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
                Allocator::default(),
            )
            .unwrap_err();
            assert_eq!(error.kind(), ErrorKind::InvalidData);
        }
    }

    #[test]
    fn compute_cache_control_requires_the_qualified_target() {
        for gfx in [120_001, 120_000] {
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
            enumerate(&root, &drm, Allocator::default(), &mut |endpoint| {
                found = Some(endpoint);
                Ok(())
            })
            .unwrap();
            let endpoint = found.unwrap();
            let queues = endpoint.gpu().unwrap().queues;
            assert!(queues.aql);
            assert_eq!(
                queues.aql_system_cache_control,
                cfg!(target_arch = "x86_64") && gfx == 120_001
            );
            assert_eq!(
                queues.pm4_system_cache_control,
                cfg!(target_arch = "x86_64") && gfx == 120_001
            );
            assert_eq!(queues.pm4, cfg!(target_arch = "x86_64") && gfx == 120_001);
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
                120_001,
                0x7550,
                268_435_456,
                Some(HostCacheability::WriteCombined),
            ),
            (120_001, 0x7550, 0, None),
            (120_500, 0x7550, 268_435_456, None),
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
            enumerate(&root, &drm, Allocator::default(), &mut |e| {
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
    fn passive_enumeration_uses_the_instance_allocator_without_global_metadata() {
        let fixture = Fixture::new();
        let (root, drm) = fixture.roots();
        let callbacks = crate::test_support::allocator::State::default();
        // SAFETY: Test state outlives the enumeration buffer and allocator.
        let allocator = unsafe { callbacks.allocator() };
        let mut endpoint = None;
        let count = crate::test_support::allocation_counter::allocations(|| {
            enumerate(&root, &drm, allocator, &mut |e| {
                endpoint = Some(e);
                Ok(())
            })
            .unwrap();
        });
        assert_eq!(count, 0);
        assert_eq!(callbacks.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(callbacks.frees.load(Ordering::Relaxed), 1);
        let session =
            crate::session::Session::new(crate::session::SessionLifetime::Session).unwrap();
        assert_eq!(
            session.activate(&endpoint.unwrap()).err().unwrap().kind(),
            ErrorKind::InvalidArgument
        );
        callbacks.fail.store(true, Ordering::Relaxed);
        let mut visited = false;
        assert_eq!(
            enumerate(&root, &drm, allocator, &mut |_| {
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
            enumerate(&root, &drm, Allocator::default(), &mut |_| {
                visited = true;
                Ok(())
            })
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidData
        );
        assert!(!visited);
    }
}
