// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! One passive, published view of independently reported topology facts.
//!
//! Node IDs are indices into this snapshot, never native IDs. A KFD node
//! ordinal and a Linux NUMA ID are separate namespaces even when equal.
//! Routes are directed reports by one source, not a physical switch graph or
//! permission to map memory. KFD sources have a checked native generation;
//! the host OS source is best effort and has no atomic generation with KFD.

use super::host::HostTopologySnapshot;
use super::{CacheInfo, Endpoint, MemoryLinkInfo};
use crate::driver::DriverInstanceId;
use crate::memory::storage::TryPush;
use crate::{Error, ErrorKind};

/// Index valid only in the snapshot that returned it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NodeId(pub usize);

/// Index valid only in the snapshot that returned it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SourceId(pub usize);

/// Source of a group of independently observed topology records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SourceKind {
    /// The Linux host OS CPU and NUMA reader.
    HostOs,
    /// One installed Linux KFD driver instance.
    LinuxKfd,
    /// Another installed activation driver.
    Driver,
}

/// Freshness and completeness of one contributor's records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopologySource {
    /// Contributor family; source IDs distinguish instances of one family.
    pub kind: SourceKind,
    /// Native generation checked around this contributor's scan, if any.
    pub generation: Option<u64>,
    /// Whether this contributor read all of its advertised records.
    pub complete: bool,
    /// Activation owner, absent for the passive host OS contributor.
    pub(crate) driver_instance: Option<DriverInstanceId>,
}

/// Native node identities remain qualified by their source and type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeNodeId {
    /// Host-wide CPU/cache observation.
    HostSystem,
    /// OS NUMA node number.
    LinuxNuma(u32),
    /// KFD topology node ordinal.
    LinuxKfd(u32),
    /// Opaque ordinal reported by another activation driver.
    Driver(u64),
}

/// Classification of a source node, independent of activated devices.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeKind {
    /// Host-wide OS observation.
    HostSystem,
    /// OS NUMA domain.
    HostNuma,
    /// KFD node with no GPU ID.
    KfdHost,
    /// KFD node reporting a GPU ID.
    KfdGpu,
    /// Node reported by another activation driver.
    Other,
}

/// One source-qualified topology node.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopologyNode {
    /// Snapshot-scoped node index.
    pub id: NodeId,
    /// Source that reported this node.
    pub source: SourceId,
    /// Source-qualified native node name.
    pub native_id: NativeNodeId,
    /// Source node classification.
    pub kind: NodeKind,
}

/// An endpoint can attach to multiple nodes; several endpoints may attach to
/// the same node. This record does not own an activated device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AttachmentKind {
    /// Activation driver node that reported the endpoint.
    Native,
    /// Best-effort host NUMA locality reported by the OS.
    HostLocality,
}

/// One typed endpoint-to-node association.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EndpointAttachment {
    /// Passive endpoint ID; it does not name a device handle.
    pub endpoint_id: [u8; 16],
    /// Node on which this endpoint was reported.
    pub node: NodeId,
    /// Meaning of this node association.
    pub kind: AttachmentKind,
}

/// Directed native route reported by one contributor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ReportedRoute {
    /// Origin node of this directed report.
    pub source: NodeId,
    /// Destination node of this directed report.
    pub target: NodeId,
    /// Native latency, bandwidth, and link-type facts.
    pub info: MemoryLinkInfo,
}

/// Cache associated with a source node, not an activated device.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TopologyCache {
    /// Node at which the source reported the cache.
    pub node: NodeId,
    /// Passive cache size, level, line size, and kind.
    pub info: CacheInfo,
}

/// Passive placement class for a reported memory capacity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryLocationKind {
    /// Host system memory reported by the OS.
    HostSystem,
    /// Separate local memory reported by an activation driver.
    DeviceLocal,
}

/// Memory capacity attached to a node, without access authorization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLocation {
    /// Source node at which the memory was reported.
    pub node: NodeId,
    /// Host or device-local placement classification.
    pub kind: MemoryLocationKind,
    /// Total bytes, when the contributor reported a capacity.
    pub capacity_bytes: Option<u64>,
    /// Host-visible subset for device-local memory, when reported.
    pub host_visible_bytes: Option<u64>,
}

/// Complete candidate published only after all driver contributions validate.
/// Host facts can be incomplete; their per-source `complete` flag records that.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopologySnapshot {
    /// Best-effort OS host observations used for CPU frontend policy.
    pub host: HostTopologySnapshot,
    /// Passive activation endpoints discovered by installed drivers.
    pub endpoints: Vec<Endpoint>,
    /// Freshness and completeness of each independent contributor.
    pub sources: Vec<TopologySource>,
    /// Source-qualified nodes, including nodes without an endpoint.
    pub nodes: Vec<TopologyNode>,
    /// Reported endpoint-to-node associations.
    pub attachments: Vec<EndpointAttachment>,
    /// Directed native routes; no access permission is implied.
    pub routes: Vec<ReportedRoute>,
    /// Caches attached to source-qualified nodes.
    pub caches: Vec<TopologyCache>,
    /// Memory locations attached to source-qualified nodes.
    pub memory_locations: Vec<MemoryLocation>,
}

impl TopologySnapshot {
    pub(crate) fn new(host: HostTopologySnapshot) -> Result<Self, Error> {
        let mut result = Self {
            host,
            endpoints: Vec::new(),
            sources: Vec::new(),
            nodes: Vec::new(),
            attachments: Vec::new(),
            routes: Vec::new(),
            caches: Vec::new(),
            memory_locations: Vec::new(),
        };
        let complete = result.host.domains_complete
            && result.host.caches.complete
            && result
                .host
                .domains
                .iter()
                .all(|domain| domain.caches.complete);
        let source = result.add_source(SourceKind::HostOs, None, complete, None)?;
        let system = result.add_node(source, NativeNodeId::HostSystem, NodeKind::HostSystem)?;
        result.memory_locations.try_push(MemoryLocation {
            node: system,
            kind: MemoryLocationKind::HostSystem,
            capacity_bytes: result.host.memory_bytes.map(|bytes| bytes as u64),
            host_visible_bytes: None,
        })?;
        for cache in &result.host.caches.records {
            result.caches.try_push(TopologyCache {
                node: system,
                info: cache.info,
            })?;
        }
        for domain in &result.host.domains {
            if result.nodes.iter().any(|node| {
                node.source == source && node.native_id == NativeNodeId::LinuxNuma(domain.numa_node)
            }) {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidData,
                    detail: "host source reported a duplicate NUMA node",
                });
            }
            let node = NodeId(result.nodes.len());
            result.nodes.try_push(TopologyNode {
                id: node,
                source,
                native_id: NativeNodeId::LinuxNuma(domain.numa_node),
                kind: NodeKind::HostNuma,
            })?;
            for cache in &domain.caches.records {
                result.caches.try_push(TopologyCache {
                    node,
                    info: cache.info,
                })?;
            }
        }
        Ok(result)
    }

    pub(crate) fn add_source(
        &mut self,
        kind: SourceKind,
        generation: Option<u64>,
        complete: bool,
        driver_instance: Option<DriverInstanceId>,
    ) -> Result<SourceId, Error> {
        if (kind == SourceKind::HostOs) != driver_instance.is_none() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidData,
                detail: "topology source has an invalid activation owner",
            });
        }
        let id = SourceId(self.sources.len());
        self.sources.try_push(TopologySource {
            kind,
            generation,
            complete,
            driver_instance,
        })?;
        Ok(id)
    }

    pub(crate) fn add_node(
        &mut self,
        source: SourceId,
        native_id: NativeNodeId,
        kind: NodeKind,
    ) -> Result<NodeId, Error> {
        let valid_type = self.sources.get(source.0).is_some_and(|source| {
            matches!(
                (source.kind, native_id, kind),
                (
                    SourceKind::HostOs,
                    NativeNodeId::HostSystem,
                    NodeKind::HostSystem
                ) | (
                    SourceKind::HostOs,
                    NativeNodeId::LinuxNuma(_),
                    NodeKind::HostNuma
                ) | (
                    SourceKind::LinuxKfd,
                    NativeNodeId::LinuxKfd(_),
                    NodeKind::KfdHost | NodeKind::KfdGpu
                ) | (SourceKind::Driver, NativeNodeId::Driver(_), NodeKind::Other)
            )
        });
        if !valid_type
            || self
                .nodes
                .iter()
                .any(|node| node.source == source && node.native_id == native_id)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidData,
                detail: "topology source reported a duplicate or invalid node",
            });
        }
        let id = NodeId(self.nodes.len());
        self.nodes.try_push(TopologyNode {
            id,
            source,
            native_id,
            kind,
        })?;
        Ok(id)
    }

    pub(crate) fn add_endpoint(&mut self, endpoint: Endpoint, node: NodeId) -> Result<(), Error> {
        if self.nodes.get(node.0).is_none()
            || self.sources[self.nodes[node.0].source.0].driver_instance
                != Some(endpoint.driver_instance)
            || self.endpoints.iter().any(|record| record.id == endpoint.id)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidData,
                detail: "topology endpoint has a duplicate ID or invalid attachment",
            });
        }
        self.attachments.try_push(EndpointAttachment {
            endpoint_id: endpoint.id,
            node,
            kind: AttachmentKind::Native,
        })?;
        self.endpoints.try_push(endpoint)?;
        Ok(())
    }

    pub(crate) fn attach_endpoint(
        &mut self,
        endpoint_id: [u8; 16],
        driver_instance: DriverInstanceId,
        node: NodeId,
        kind: AttachmentKind,
    ) -> Result<(), Error> {
        if self.nodes.get(node.0).is_none()
            || !self
                .endpoints
                .iter()
                .any(|record| record.id == endpoint_id && record.driver_instance == driver_instance)
            || self.attachments.iter().any(|attachment| {
                attachment.endpoint_id == endpoint_id
                    && attachment.node == node
                    && attachment.kind == kind
            })
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidData,
                detail: "topology attachment is duplicate or names an unknown record",
            });
        }
        self.attachments.try_push(EndpointAttachment {
            endpoint_id,
            node,
            kind,
        })?;
        Ok(())
    }

    pub(crate) fn add_route(
        &mut self,
        source: NodeId,
        target: NodeId,
        info: MemoryLinkInfo,
    ) -> Result<(), Error> {
        let from = self.nodes.get(source.0);
        let to = self.nodes.get(target.0);
        if !matches!((from, to), (Some(a), Some(b)) if a.source == b.source) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidData,
                detail: "reported route crosses topology sources or has an invalid node",
            });
        }
        if let Some(route) = self
            .routes
            .iter_mut()
            .find(|route| route.source == source && route.target == target)
        {
            route.info = info;
            return Ok(());
        }
        self.routes.try_push(ReportedRoute {
            source,
            target,
            info,
        })?;
        Ok(())
    }

    pub(crate) fn add_memory_location(&mut self, location: MemoryLocation) -> Result<(), Error> {
        if self.nodes.get(location.node.0).is_none()
            || location
                .host_visible_bytes
                .zip(location.capacity_bytes)
                .is_some_and(|(visible, total)| visible > total)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidData,
                detail: "topology memory location is inconsistent or has no node",
            });
        }
        if let Some(existing) = self
            .memory_locations
            .iter()
            .find(|existing| existing.node == location.node && existing.kind == location.kind)
        {
            return if *existing == location {
                Ok(())
            } else {
                Err(Error::Operation {
                    kind: ErrorKind::InvalidData,
                    detail: "topology source reported conflicting memory capacities",
                })
            };
        }
        self.memory_locations.try_push(location)?;
        Ok(())
    }

    /// Nodes to which the endpoint is attached in this snapshot.
    pub fn nodes_for_endpoint(&self, endpoint: &Endpoint) -> impl Iterator<Item = &TopologyNode> {
        let owned = self.endpoints.iter().any(|candidate| {
            candidate.id == endpoint.id && candidate.driver_instance == endpoint.driver_instance
        });
        self.attachments
            .iter()
            .filter(move |attachment| owned && attachment.endpoint_id == endpoint.id)
            .filter_map(|attachment| self.nodes.get(attachment.node.0))
    }

    fn native_nodes_for_endpoint(
        &self,
        endpoint: &Endpoint,
    ) -> impl Iterator<Item = &TopologyNode> {
        self.nodes_for_endpoint(endpoint).filter(|node| {
            self.attachments.iter().any(|attachment| {
                attachment.endpoint_id == endpoint.id
                    && attachment.node == node.id
                    && attachment.kind == AttachmentKind::Native
            })
        })
    }

    /// Cache records reported by the endpoint's native activation nodes.
    /// Host-locality attachments do not make host caches into GPU caches.
    pub fn caches_for_endpoint(&self, endpoint: &Endpoint) -> impl Iterator<Item = &CacheInfo> {
        self.caches.iter().filter_map(move |cache| {
            self.native_nodes_for_endpoint(endpoint)
                .any(|node| node.id == cache.node)
                .then_some(&cache.info)
        })
    }

    /// Reported directed route between two endpoint attachments.
    #[must_use]
    pub fn route_between(&self, from: &Endpoint, to: &Endpoint) -> Option<MemoryLinkInfo> {
        self.routes.iter().rev().find_map(|route| {
            let source = self.nodes.get(route.source.0)?;
            let target = self.nodes.get(route.target.0)?;
            (source.source == target.source
                && self
                    .native_nodes_for_endpoint(from)
                    .any(|node| node.id == source.id)
                && self
                    .native_nodes_for_endpoint(to)
                    .any(|node| node.id == target.id))
            .then_some(route.info)
        })
    }

    /// Reported directed route from an endpoint to a KFD host node.
    #[must_use]
    pub fn route_to_host(&self, from: &Endpoint) -> Option<MemoryLinkInfo> {
        self.routes.iter().rev().find_map(|route| {
            let source = self.nodes.get(route.source.0)?;
            let target = self.nodes.get(route.target.0)?;
            (source.source == target.source
                && target.kind == NodeKind::KfdHost
                && self
                    .native_nodes_for_endpoint(from)
                    .any(|node| node.id == source.id))
            .then_some(route.info)
        })
    }

    /// Reported directed route from a KFD host node to an endpoint.
    #[must_use]
    pub fn route_from_host(&self, to: &Endpoint) -> Option<MemoryLinkInfo> {
        self.routes.iter().rev().find_map(|route| {
            let source = self.nodes.get(route.source.0)?;
            let target = self.nodes.get(route.target.0)?;
            (source.source == target.source
                && source.kind == NodeKind::KfdHost
                && self
                    .native_nodes_for_endpoint(to)
                    .any(|node| node.id == target.id))
            .then_some(route.info)
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::{EndpointSelector, new_driver_instance};
    use crate::memory::DeviceAccess;
    use crate::topology::host::{HostCaches, HostDomain};
    use crate::topology::{CacheKind, EndpointKind};

    fn cache(level: u32) -> CacheInfo {
        CacheInfo::new(level, 4096, Some(64), CacheKind::Unified)
    }

    fn host() -> HostTopologySnapshot {
        HostTopologySnapshot {
            memory_bytes: None,
            cpu: None,
            online_cpus: None,
            caches: HostCaches {
                records: Vec::new(),
                complete: false,
            },
            domains: vec![HostDomain {
                numa_node: 7,
                cpus: None,
                caches: HostCaches {
                    records: Vec::new(),
                    complete: true,
                },
            }],
            domains_complete: true,
        }
    }

    fn endpoint(id: u8) -> Endpoint {
        Endpoint {
            id: [id; 16],
            name: [0; 128],
            product_name: None,
            pci: None,
            kind: EndpointKind::Npu,
            local_memory_bytes: 0,
            host_visible_local_memory_bytes: 0,
            host_local_cacheability: None,
            allocation_granularity: 4096,
            address_bit_count: 48,
            minimum_address: 0,
            maximum_address: u64::MAX,
            supported_permissions: DeviceAccess::NONE,
            driver_instance: new_driver_instance(),
            selector: EndpointSelector::Opaque(u64::from(id)),
        }
    }

    fn link() -> MemoryLinkInfo {
        MemoryLinkInfo {
            minimum_latency: 1,
            maximum_latency: 2,
            minimum_bandwidth: 3,
            maximum_bandwidth: 4,
            atomic_support_32bit: false,
            atomic_support_64bit: false,
            coherent_support: false,
            kind: super::super::MemoryLinkType::Pcie,
            numa_distance: 20,
        }
    }

    #[test]
    fn source_qualified_nodes_allow_multiple_endpoints_and_typed_attachments() {
        let mut snapshot = TopologySnapshot::new(host()).unwrap();
        let first = endpoint(1);
        let mut second = endpoint(2);
        second.driver_instance = first.driver_instance;
        let source = snapshot
            .add_source(
                SourceKind::LinuxKfd,
                Some(91),
                true,
                Some(first.driver_instance),
            )
            .unwrap();
        let gpu = snapshot
            .add_node(source, NativeNodeId::LinuxKfd(7), NodeKind::KfdGpu)
            .unwrap();
        let host_numa = snapshot
            .nodes
            .iter()
            .find(|node| node.native_id == NativeNodeId::LinuxNuma(7))
            .unwrap()
            .id;
        assert_ne!(gpu, host_numa);
        assert_eq!(snapshot.nodes[gpu.0].source, source);
        assert_eq!(snapshot.nodes[host_numa.0].source, SourceId(0));
        assert_eq!(snapshot.sources[0].generation, None);
        assert!(!snapshot.sources[0].complete);
        assert_eq!(snapshot.sources[source.0].generation, Some(91));
        snapshot.add_endpoint(first.clone(), gpu).unwrap();
        snapshot.add_endpoint(second.clone(), gpu).unwrap();
        snapshot
            .attach_endpoint(
                first.id,
                first.driver_instance,
                host_numa,
                AttachmentKind::HostLocality,
            )
            .unwrap();
        snapshot.caches.push(TopologyCache {
            node: gpu,
            info: cache(2),
        });
        snapshot.caches.push(TopologyCache {
            node: host_numa,
            info: cache(3),
        });
        let location = MemoryLocation {
            node: gpu,
            kind: MemoryLocationKind::DeviceLocal,
            capacity_bytes: Some(8192),
            host_visible_bytes: Some(4096),
        };
        snapshot.add_memory_location(location).unwrap();
        snapshot.add_memory_location(location).unwrap();
        assert_eq!(snapshot.memory_locations.len(), 2);
        assert_eq!(snapshot.memory_locations[1], location);
        assert_eq!(
            snapshot
                .add_memory_location(MemoryLocation {
                    capacity_bytes: Some(16384),
                    ..location
                })
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(snapshot.nodes_for_endpoint(&first).count(), 2);
        assert_eq!(snapshot.nodes_for_endpoint(&second).count(), 1);
        assert_eq!(snapshot.caches_for_endpoint(&first).count(), 1);
        assert_eq!(
            snapshot.caches_for_endpoint(&first).next().unwrap().level,
            2
        );
        let mut foreign = first.clone();
        foreign.driver_instance = new_driver_instance();
        assert_eq!(snapshot.nodes_for_endpoint(&foreign).count(), 0);
    }

    #[test]
    fn routes_are_directed_reports_with_no_cross_source_inference() {
        let mut snapshot = TopologySnapshot::new(host()).unwrap();
        let first = endpoint(1);
        let mut second = endpoint(2);
        second.driver_instance = first.driver_instance;
        let source = snapshot
            .add_source(
                SourceKind::LinuxKfd,
                Some(8),
                true,
                Some(first.driver_instance),
            )
            .unwrap();
        let from = snapshot
            .add_node(source, NativeNodeId::LinuxKfd(1), NodeKind::KfdGpu)
            .unwrap();
        let to = snapshot
            .add_node(source, NativeNodeId::LinuxKfd(2), NodeKind::KfdGpu)
            .unwrap();
        let host_node = snapshot
            .add_node(source, NativeNodeId::LinuxKfd(0), NodeKind::KfdHost)
            .unwrap();
        snapshot.add_endpoint(first.clone(), from).unwrap();
        snapshot.add_endpoint(second.clone(), to).unwrap();
        snapshot.add_route(from, to, link()).unwrap();
        snapshot.add_route(host_node, from, link()).unwrap();
        assert_eq!(snapshot.route_between(&first, &second), Some(link()));
        assert_eq!(snapshot.route_between(&second, &first), None);
        assert_eq!(snapshot.route_from_host(&first), Some(link()));
        assert_eq!(snapshot.route_to_host(&first), None);
        assert_eq!(
            snapshot
                .add_route(from, NodeId(1), link())
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
    }
}
