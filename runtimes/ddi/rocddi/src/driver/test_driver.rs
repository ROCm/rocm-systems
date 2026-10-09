// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Minimal non-GPU driver used to exercise heterogeneous session routing.

use super::{AddressSpaceInfo, Driver, EndpointSelector};
use crate::host_storage::Allocator;
use crate::memory::DeviceAccess;
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, EndpointKind, TopologyKey};
use crate::{Error, ErrorKind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Test driver instance with only the common driver contract.
pub(crate) struct TestDriver {
    instance: u64,
    id: [u8; 16],
    kind: EndpointKind,
    lifetime: DriverContextLifetime,
    pub(crate) shutdowns: Arc<AtomicUsize>,
    pub(crate) fail_shutdown: Arc<AtomicBool>,
}

/// Activated test endpoint. It intentionally has no GPU resource capabilities.
#[derive(Clone)]
pub(crate) struct TestDevice {
    id: [u8; 16],
}

impl TestDriver {
    pub(crate) fn new(id: [u8; 16], kind: EndpointKind, lifetime: DriverContextLifetime) -> Self {
        Self {
            instance: super::new_driver_instance(),
            id,
            kind,
            lifetime,
            shutdowns: Arc::new(AtomicUsize::new(0)),
            fail_shutdown: Arc::new(AtomicBool::new(false)),
        }
    }

    fn endpoint(&self) -> Endpoint {
        let mut name = [0; 128];
        name[..8].copy_from_slice(b"test-npu");
        Endpoint {
            id: self.id,
            name,
            pci: None,
            kind: self.kind,
            local_memory_bytes: 0,
            host_visible_local_memory_bytes: 0,
            host_local_cacheability: None,
            allocation_granularity: 4096,
            address_bit_count: 48,
            minimum_address: 0x1000,
            maximum_address: 0xffff,
            supported_permissions: DeviceAccess::NONE,
            caches: None,
            memory_links: None,
            topology_key: TopologyKey {
                group: 0,
                member: 0,
            },
            driver_instance: self.instance,
            selector: EndpointSelector::Opaque(u64::from_le_bytes([
                self.id[0], self.id[1], self.id[2], self.id[3], self.id[4], self.id[5], self.id[6],
                self.id[7],
            ])),
        }
    }
}

impl AddressSpaceInfo for TestDevice {
    fn address_range(&self) -> (u64, u64) {
        (0x1000, 0xffff)
    }

    fn shares_address_domain(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Driver for TestDriver {
    type DeviceState = TestDevice;

    fn allocator(&self) -> Allocator {
        Allocator::system()
    }

    fn driver_instance(&self) -> u64 {
        self.instance
    }

    fn context_lifetime(&self) -> DriverContextLifetime {
        self.lifetime
    }

    fn shutdown(&mut self) -> Result<(), Error> {
        self.shutdowns.fetch_add(1, Ordering::Relaxed);
        if self.fail_shutdown.swap(false, Ordering::Relaxed) {
            return Err(Error::Operation {
                kind: ErrorKind::Driver,
                detail: "injected shutdown failure",
            });
        }
        Ok(())
    }

    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        visitor(self.endpoint())
    }

    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        if id != self.id {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "unknown test endpoint",
            });
        }
        Ok(self.endpoint())
    }

    fn activate(&self, endpoint: &Endpoint) -> Result<TestDevice, Error> {
        if endpoint != &self.endpoint() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "foreign test endpoint",
            });
        }
        Ok(TestDevice { id: self.id })
    }
}
