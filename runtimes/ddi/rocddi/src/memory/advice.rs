// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Provider-qualified policy and migration for process virtual-address ranges.
//!
//! A service retains exactly one provider. Device targets are created only from
//! activated devices and carry that provider's identity. Range policy is not a
//! backing owner or a residency lease; callers retain their own memory lifetime.
//! New policy, query, and prefetch transactions require session admission.
//! Discarding host backing remains available while retiring the session.

use crate::device::Device;
use crate::driver::{Driver, DriverInstance, DriverInstanceId, MemoryAdviceOperations};
use crate::error::error;
use crate::memory::storage::Shared;
use crate::session::Session;
use crate::{Error, ErrorKind};

/// Activated endpoint identity qualified by the exact provider instance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdviceTarget {
    pub(crate) provider: DriverInstanceId,
    pub(crate) endpoint: [u8; 16],
}

impl AdviceTarget {
    /// Returns the passive endpoint identity for frontend agent lookup.
    #[must_use]
    pub fn endpoint_id(self) -> [u8; 16] {
        self.endpoint
    }
}

/// Physical placement of a process range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceLocation {
    /// Host system memory.
    System,
    /// No uniform location across the range, or no location specified.
    UnspecifiedOrMixed,
    /// Memory associated with an activated endpoint.
    Device(AdviceTarget),
}

/// Per-device access policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceAccess {
    /// Faulting access with migration permitted.
    Accessible,
    /// Access without migration.
    AccessibleInPlace,
    /// Access denied.
    NoAccess,
}

/// Independently settable range properties, with no native flag encoding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceProperty {
    /// Allow host access.
    HostAccess,
    /// Coherent fine-grained access.
    Coherent,
    /// Restrict preferred placement to the local device hive.
    HiveLocal,
    /// Device access is read-only.
    DeviceReadOnly,
    /// Allow device instruction fetch.
    DeviceExecutable,
    /// Optimize for mostly device reads.
    DeviceReadMostly,
}

/// Reduction of a property over every page in a queried range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceState {
    /// Enabled throughout the range.
    Enabled,
    /// Disabled throughout the range.
    Disabled,
    /// Neither uniformly enabled nor uniformly disabled.
    Mixed,
}

/// One policy mutation. Prefetch has a separate operation and lifetime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceChange {
    /// Set or clear a semantic property.
    Property(AdviceProperty, bool),
    /// Set or remove a preferred location.
    PreferredLocation(AdviceLocation),
    /// Change an activated device's access.
    Access(AdviceTarget, AdviceAccess),
    /// Migration granularity in bytes, a power-of-two multiple of the base unit.
    /// This can exceed the address-space extent; it describes policy units.
    MigrationGranularity(u128),
}

/// One query, with no output storage reused as native input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceQuery {
    /// Reduce a property over the range.
    Property(AdviceProperty),
    /// Uniform preferred location, when present.
    PreferredLocation,
    /// Uniform last prefetch location, when present.
    PrefetchLocation,
    /// Device access policy.
    Access(AdviceTarget),
    /// Minimum migration granularity over the range, in bytes.
    MigrationGranularity,
}

/// One staged response corresponding to the query at the same index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdviceValue {
    /// Uniform or mixed property state.
    Property(AdviceState),
    /// Preferred or last prefetch location.
    Location(AdviceLocation),
    /// Access policy for the queried target.
    Access(AdviceAccess),
    /// Minimum migration granularity, in bytes.
    MigrationGranularity(u128),
}

/// Required strength of a prefetch operation's successful return.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefetchCompletion {
    /// A best-effort native attempt returned; placement may have failed or
    /// fallen back. This cannot satisfy a migration-complete API contract.
    Attempt,
    /// The full requested migration completed; later eviction is still possible.
    Migration,
}

/// Provider constraints; these do not reserve memory or promise residency.
#[derive(Clone, Copy, Debug)]
pub struct AdviceCapabilities {
    /// Strongest prefetch completion guarantee qualified by this provider.
    pub prefetch_completion: PrefetchCompletion,
    /// Required address and size alignment.
    pub range_alignment: u64,
    /// Smallest supported migration granularity in bytes.
    pub minimum_migration_granularity: u128,
    /// Largest supported migration granularity in bytes.
    pub maximum_migration_granularity: u128,
}

/// Concrete retained providers for the optional memory-advice facet.
enum Provider {
    #[cfg(target_os = "linux")]
    Kfd(Shared<crate::driver::KfdDriver>),
    #[cfg(test)]
    Test(Shared<crate::driver::test_driver::TestDriver>),
    #[cfg(not(target_os = "linux"))]
    Unavailable(std::convert::Infallible),
}

/// One retained provider service, safe to move to a frontend worker.
pub struct MemoryAdvice {
    provider: Provider,
    process: crate::os::ProcessIdentity,
}

impl Session {
    /// Selects the only capable provider in this session.
    ///
    /// # Errors
    /// Rejects zero providers or ambiguity; never selects the first provider.
    pub fn memory_advice(&self) -> Result<MemoryAdvice, Error> {
        let mut selected = None;
        for driver in self.installed_drivers()? {
            if let Some(provider) = Provider::from_driver(driver) {
                if selected.is_some() {
                    return Err(error(
                        ErrorKind::InvalidArgument,
                        "memory advice requires an explicit provider",
                    ));
                }
                selected = Some(provider);
            }
        }
        let provider = selected.ok_or_else(|| {
            error(
                ErrorKind::Unsupported,
                "session has no memory-advice provider",
            )
        })?;
        let _operation = provider.admit()?;
        Ok(MemoryAdvice {
            provider,
            process: self.process_identity(),
        })
    }

    /// Selects the exact provider owning an activated device.
    ///
    /// # Errors
    /// Rejects a foreign device or a provider without this capability.
    pub fn memory_advice_for(&self, device: &Device) -> Result<MemoryAdvice, Error> {
        let driver = self
            .installed_drivers()?
            .iter()
            .find(|driver| driver.instance_id() == device.endpoint.driver_instance)
            .ok_or_else(|| {
                error(
                    ErrorKind::InvalidArgument,
                    "memory-advice device belongs to another session",
                )
            })?;
        let provider = Provider::from_driver(driver).ok_or_else(|| {
            error(
                ErrorKind::Unsupported,
                "device provider has no memory advice",
            )
        })?;
        let _operation = provider.admit()?;
        Ok(MemoryAdvice {
            provider,
            process: self.process_identity(),
        })
    }
}

// Concrete routing retains associated provider types without imposing an OS
// trait object or adding memory-advice requirements to the base Driver trait.
macro_rules! route {
    ($self:expr, $driver:ident => $operation:expr) => {
        match $self {
            #[cfg(target_os = "linux")]
            Provider::Kfd($driver) => $operation,
            #[cfg(test)]
            Provider::Test($driver) => $operation,
            #[cfg(not(target_os = "linux"))]
            Provider::Unavailable(never) => match *never {},
        }
    };
}

impl Provider {
    fn admit(&self) -> Result<crate::session::SessionOperation, Error> {
        route!(self, driver => driver.admission().enter())
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "optional facets differ between installed providers"
    )]
    fn from_driver(driver: &DriverInstance) -> Option<Self> {
        match driver {
            #[cfg(target_os = "linux")]
            DriverInstance::LinuxKfd(driver) => Some(Self::Kfd(driver.clone())),
            #[cfg(test)]
            DriverInstance::Test(driver) => {
                driver.advice_enabled.then(|| Self::Test(driver.clone()))
            }
            #[cfg(not(target_os = "linux"))]
            DriverInstance::Unavailable(never) => match *never {},
        }
    }
    fn query(
        &self,
        address: u64,
        size: u64,
        queries: &[AdviceQuery],
    ) -> Result<Vec<AdviceValue>, Error> {
        route!(self, driver => driver.advice_query(address, size, queries))
    }

    fn identity(&self) -> DriverInstanceId {
        route!(self, driver => driver.driver_instance())
    }
}

impl MemoryAdvice {
    /// Enables demand-paged host access to an owned anonymous reservation.
    /// The operation does not register a device mapping or set SVM policy.
    ///
    /// # Safety
    /// The caller owns this complete anonymous range, excludes overlapping
    /// native mappings and protection changes, and retains it through return.
    ///
    /// # Errors
    /// Rejects invalid ranges, an inherited service, or unsupported host services.
    #[allow(unsafe_code)]
    pub unsafe fn commit_host_range(&self, address: u64, size: u64) -> Result<(), Error> {
        let _operation = self.provider.admit()?;
        self.check_host_range(address, size)?;
        #[cfg(target_os = "linux")]
        {
            let address = usize::try_from(address)
                .map_err(|_| error(ErrorKind::InvalidArgument, "SVM address exceeds host width"))?;
            let size = usize::try_from(size)
                .map_err(|_| error(ErrorKind::InvalidArgument, "SVM size exceeds host width"))?;
            // SAFETY: The caller retains the entire anonymous reservation.
            unsafe { crate::os::linux::memory::commit_svm_range(address, size) }
                .map_err(|source| crate::os::native_error("commit SVM host range", source))
        }
        #[cfg(not(target_os = "linux"))]
        Err(error(
            ErrorKind::Unsupported,
            "SVM host backing is unavailable",
        ))
    }

    /// Discards anonymous host pages after any required device migration.
    /// With `immediate` false, Linux may reclaim the contents under pressure;
    /// with it true, later host reads observe fresh zero-filled pages.
    ///
    /// # Safety
    /// The caller retains this anonymous, host-accessible mapping and excludes
    /// host/device accesses whose data must survive. Device users must have
    /// retired or migrated before their physical pages can be discarded.
    ///
    /// # Errors
    /// Rejects invalid ranges or unavailable native discard; propagates OS errors.
    #[allow(unsafe_code)]
    pub unsafe fn discard_host_range(
        &self,
        address: u64,
        size: u64,
        immediate: bool,
    ) -> Result<(), Error> {
        self.check_host_range(address, size)?;
        #[cfg(target_os = "linux")]
        {
            let address = usize::try_from(address)
                .map_err(|_| error(ErrorKind::InvalidArgument, "SVM address exceeds host width"))?;
            let size = usize::try_from(size)
                .map_err(|_| error(ErrorKind::InvalidArgument, "SVM size exceeds host width"))?;
            // SAFETY: The caller retains and synchronizes the anonymous mapping.
            unsafe { crate::os::linux::memory::discard_svm_range(address, size, immediate) }
                .map_err(|source| crate::os::native_error("discard SVM host range", source))
        }
        #[cfg(not(target_os = "linux"))]
        Err(error(
            ErrorKind::Unsupported,
            "SVM host discard is unavailable",
        ))
    }
    fn check(&self) -> Result<(), Error> {
        self.process.check().map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "memory advice belongs to another process",
            )
        })
    }

    /// Returns this provider's range and migration units.
    ///
    /// # Errors
    /// Reports provider unavailability or failure to query its constraints.
    pub fn capabilities(&self) -> Result<AdviceCapabilities, Error> {
        let _operation = self.provider.admit()?;
        self.provider_capabilities()
    }

    fn provider_capabilities(&self) -> Result<AdviceCapabilities, Error> {
        self.check()?;
        route!(&self.provider, driver => driver.advice_capabilities())
    }

    /// Qualifies a device for use in this service's requests.
    ///
    /// # Errors
    /// Rejects devices from another provider, even with the same endpoint ID.
    pub fn target(&self, device: &Device) -> Result<AdviceTarget, Error> {
        self.check()?;
        let target = AdviceTarget {
            provider: device.endpoint.driver_instance,
            endpoint: device.endpoint.id,
        };
        self.check_target(target)?;
        Ok(target)
    }

    fn check_target(&self, target: AdviceTarget) -> Result<(), Error> {
        if target.provider != self.provider.identity() {
            return Err(error(
                ErrorKind::InvalidArgument,
                "memory-advice target belongs to another provider",
            ));
        }
        Ok(())
    }

    fn check_location(&self, location: AdviceLocation) -> Result<(), Error> {
        if let AdviceLocation::Device(target) = location {
            self.check_target(target)?;
        }
        Ok(())
    }

    // Host backing has only host page constraints. It must work before GPU
    // activation, and discard remains a retirement operation after admission closes.
    fn check_host_range(&self, address: u64, size: u64) -> Result<(), Error> {
        self.check()?;
        Self::validate_range(address, size, crate::memory::host_page_size()?)
    }

    fn validate_range(address: u64, size: u64, alignment: u64) -> Result<(), Error> {
        if address == 0
            || size == 0
            || address.checked_add(size).is_none()
            || address % alignment != 0
            || size % alignment != 0
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid memory-advice range",
            ));
        }
        Ok(())
    }

    fn check_range(&self, address: u64, size: u64) -> Result<AdviceCapabilities, Error> {
        let caps = self.provider_capabilities()?;
        if !caps.range_alignment.is_power_of_two()
            || !caps.minimum_migration_granularity.is_power_of_two()
            || !caps.maximum_migration_granularity.is_power_of_two()
            || caps.maximum_migration_granularity < caps.minimum_migration_granularity
        {
            return Err(error(
                ErrorKind::DriverContract,
                "invalid memory-advice range alignment",
            ));
        }
        Self::validate_range(address, size, caps.range_alignment)?;
        Ok(caps)
    }

    /// Applies a fully validated mutation batch to one provider.
    ///
    /// Validation fails before native effects. A native failure need not roll
    /// back policy already applied; callers must not blindly replay the batch.
    ///
    /// # Errors
    /// Rejects malformed ranges, foreign targets, unsupported units or native errors.
    pub fn set(&self, address: u64, size: u64, changes: &[AdviceChange]) -> Result<(), Error> {
        let _operation = self.provider.admit()?;
        let caps = self.check_range(address, size)?;
        for change in changes {
            match *change {
                AdviceChange::PreferredLocation(location) => self.check_location(location)?,
                AdviceChange::Access(target, _) => self.check_target(target)?,
                AdviceChange::MigrationGranularity(bytes)
                    if !bytes.is_power_of_two()
                        || bytes < caps.minimum_migration_granularity
                        || bytes > caps.maximum_migration_granularity =>
                {
                    return Err(error(
                        ErrorKind::InvalidArgument,
                        "unsupported migration granularity",
                    ));
                }
                _ => (),
            }
        }
        route!(&self.provider, driver => driver.advice_set(address, size, changes))
    }

    /// Queries a complete batch, publishing no partial output on failure.
    ///
    /// # Errors
    /// Rejects invalid requests, malformed provider responses or native failures.
    pub fn query(
        &self,
        address: u64,
        size: u64,
        queries: &[AdviceQuery],
    ) -> Result<Vec<AdviceValue>, Error> {
        let _operation = self.provider.admit()?;
        let caps = self.check_range(address, size)?;
        for query in queries {
            if let AdviceQuery::Access(target) = *query {
                self.check_target(target)?;
            }
        }
        let values = self.provider.query(address, size, queries)?;
        if queries.len() != values.len() {
            return Err(error(
                ErrorKind::DriverContract,
                "memory-advice response count differs",
            ));
        }
        for (query, value) in queries.iter().zip(&values) {
            match (query, value) {
                (AdviceQuery::Property(_), AdviceValue::Property(_))
                | (AdviceQuery::Access(_), AdviceValue::Access(_)) => (),
                (
                    AdviceQuery::PreferredLocation | AdviceQuery::PrefetchLocation,
                    AdviceValue::Location(location),
                ) => {
                    self.check_location(*location).map_err(|_| {
                        error(ErrorKind::DriverContract, "foreign memory-advice result")
                    })?;
                }
                (AdviceQuery::MigrationGranularity, AdviceValue::MigrationGranularity(bytes))
                    if bytes.is_power_of_two()
                        && *bytes >= caps.minimum_migration_granularity
                        && *bytes <= caps.maximum_migration_granularity => {}
                _ => {
                    return Err(error(
                        ErrorKind::DriverContract,
                        "memory-advice response type differs",
                    ));
                }
            }
        }
        Ok(values)
    }

    /// Validates a prefetch before a frontend accepts asynchronous work.
    ///
    /// # Errors
    /// Rejects invalid ranges, unspecified destinations and foreign targets.
    pub fn validate_prefetch(
        &self,
        address: u64,
        size: u64,
        location: AdviceLocation,
        completion: PrefetchCompletion,
    ) -> Result<(), Error> {
        let _operation = self.provider.admit()?;
        let caps = self.check_range(address, size)?;
        if completion == PrefetchCompletion::Migration && caps.prefetch_completion != completion {
            return Err(error(
                ErrorKind::Unsupported,
                "provider cannot prove completed migration",
            ));
        }
        self.check_location(location)?;
        if location == AdviceLocation::UnspecifiedOrMixed {
            return Err(error(
                ErrorKind::InvalidArgument,
                "prefetch requires a destination",
            ));
        }
        Ok(())
    }

    /// Issues a prefetch after frontend dependencies have completed.
    ///
    /// Success satisfies the requested completion strength. An attempt does
    /// not promise placement; even completed migration establishes no durable
    /// residency lease and does not prevent later faults or eviction.
    ///
    /// # Errors
    /// Reports invalid requests, unsupported migration or native failure.
    pub fn prefetch(
        &self,
        address: u64,
        size: u64,
        location: AdviceLocation,
        completion: PrefetchCompletion,
    ) -> Result<(), Error> {
        let _operation = self.provider.admit()?;
        self.validate_prefetch(address, size, location, completion)?;
        route!(&self.provider, driver => driver.advice_prefetch(address, size, location, completion))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::test_driver::TestDriver;
    use crate::session::DriverContextLifetime;
    use crate::topology::EndpointKind;
    use std::sync::atomic::Ordering;

    fn provider(id: u8) -> Shared<TestDriver> {
        let mut driver =
            TestDriver::new([id; 16], EndpointKind::Npu, DriverContextLifetime::Session);
        driver.advice_enabled = true;
        Shared::new(driver).unwrap()
    }

    fn session(drivers: Vec<Shared<TestDriver>>) -> Session {
        Session::from_drivers(
            drivers.into_iter().map(DriverInstance::Test).collect(),
            DriverContextLifetime::Session,
        )
        .unwrap()
    }

    fn installed_provider(session: &Session, id: u8) -> Shared<TestDriver> {
        session
            .installed_drivers()
            .unwrap()
            .iter()
            .find_map(|driver| match driver {
                DriverInstance::Test(driver) if driver.open_endpoint([id; 16]).is_ok() => {
                    Some(driver.clone())
                }
                _ => None,
            })
            .unwrap()
    }

    fn device(session: &Session, id: u8) -> Device {
        session
            .activate(&session.open_endpoint([id; 16]).unwrap())
            .unwrap()
    }

    #[cfg(target_os = "linux")]
    #[test]
    #[allow(unsafe_code)]
    fn host_backing_does_not_require_an_active_kfd_context() {
        use crate::os::linux::memory::{Reservation, page_size};

        for lifetime in [
            DriverContextLifetime::Process,
            DriverContextLifetime::Session,
        ] {
            let session = Session::new(lifetime).unwrap();
            let service = session.memory_advice().unwrap();
            assert_eq!(
                service.capabilities().unwrap().prefetch_completion,
                PrefetchCompletion::Attempt
            );
            let page = page_size().unwrap();
            let mut reservation =
                Reservation::new(page, page, (page as u64, isize::MAX as u64), false).unwrap();
            let address = reservation.address() as u64;
            // SAFETY: This test exclusively owns the anonymous reservation and
            // retains it through both host operations; no device was activated.
            unsafe {
                service.commit_host_range(address, page as u64).unwrap();
                (address as *mut u8).write(0xa5);
                service
                    .discard_host_range(address, page as u64, true)
                    .unwrap();
                assert_eq!((address as *const u8).read(), 0);
            }
            reservation.release().unwrap();
        }
    }

    #[test]
    fn a_non_kfd_provider_executes_policy_query_and_prefetch() {
        let driver = provider(1);
        let session = session(vec![driver]);
        let driver = installed_provider(&session, 1);
        let device = device(&session, 1);
        let service = session.memory_advice_for(&device).unwrap();
        let target = service.target(&device).unwrap();
        let changes = [
            AdviceChange::Property(AdviceProperty::Coherent, true),
            AdviceChange::MigrationGranularity(8192),
            AdviceChange::Access(target, AdviceAccess::Accessible),
        ];
        service.set(0x10000, 4096, &changes).unwrap();
        assert_eq!(*driver.advice_changes.lock().unwrap(), changes);
        *driver.advice_results.lock().unwrap() = vec![
            AdviceValue::Property(AdviceState::Mixed),
            AdviceValue::Location(AdviceLocation::Device(target)),
        ];
        assert_eq!(
            service
                .query(
                    0x10000,
                    4096,
                    &[
                        AdviceQuery::Property(AdviceProperty::Coherent),
                        AdviceQuery::PreferredLocation
                    ]
                )
                .unwrap(),
            *driver.advice_results.lock().unwrap()
        );
        service
            .prefetch(
                0x10000,
                4096,
                AdviceLocation::Device(target),
                PrefetchCompletion::Migration,
            )
            .unwrap();
        assert_eq!(driver.advice_calls.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn ambiguous_providers_and_same_id_foreign_devices_are_rejected() {
        let first = provider(1);
        let second = provider(2);
        let local = session(vec![first, second]);
        let first = installed_provider(&local, 1);
        let second = installed_provider(&local, 2);
        assert!(local.memory_advice().is_err());
        let service = local.memory_advice_for(&device(&local, 1)).unwrap();
        let other_service = local.memory_advice_for(&device(&local, 2)).unwrap();
        let foreign_target = other_service.target(&device(&local, 2)).unwrap();
        let foreign = session(vec![provider(1)]);
        let same_id = device(&foreign, 1);
        assert!(local.memory_advice_for(&same_id).is_err());
        assert!(service.target(&same_id).is_err());
        let changes = [
            AdviceChange::Property(AdviceProperty::Coherent, true),
            AdviceChange::Access(foreign_target, AdviceAccess::Accessible),
        ];
        assert!(service.set(0x10000, 4096, &changes).is_err());
        assert!(
            service
                .query(0x10000, 4096, &[AdviceQuery::Access(foreign_target)])
                .is_err()
        );
        assert!(
            service
                .prefetch(
                    0x10000,
                    4096,
                    AdviceLocation::Device(foreign_target),
                    PrefetchCompletion::Migration
                )
                .is_err()
        );
        assert_eq!(first.advice_calls.load(Ordering::Relaxed), 0);
        assert_eq!(second.advice_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn validation_precedes_native_effects_and_rejects_malformed_later_results() {
        let driver = provider(1);
        let session = session(vec![driver]);
        let driver = installed_provider(&session, 1);
        let service = session.memory_advice().unwrap();
        assert!(
            service
                .set(
                    0x10000,
                    4096,
                    &[
                        AdviceChange::Property(AdviceProperty::Coherent, true),
                        AdviceChange::MigrationGranularity(6000)
                    ]
                )
                .is_err()
        );
        assert!(service.set(u64::MAX - 4095, 4096, &[]).is_err());
        assert!(
            service
                .prefetch(
                    0x10000,
                    4096,
                    AdviceLocation::UnspecifiedOrMixed,
                    PrefetchCompletion::Migration
                )
                .is_err()
        );
        assert_eq!(driver.advice_calls.load(Ordering::Relaxed), 0);
        *driver.advice_results.lock().unwrap() = vec![
            AdviceValue::Property(AdviceState::Enabled),
            AdviceValue::MigrationGranularity(0),
        ];
        assert_eq!(
            service
                .query(
                    0x10000,
                    4096,
                    &[
                        AdviceQuery::Property(AdviceProperty::Coherent),
                        AdviceQuery::MigrationGranularity
                    ]
                )
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );
    }

    #[test]
    fn retirement_waits_for_admitted_prefetch_and_rejects_new_work() {
        use std::sync::atomic::AtomicBool;
        use std::sync::{Arc, Barrier};
        let gate = Arc::new((Barrier::new(2), Barrier::new(2)));
        let completed = Arc::new(AtomicBool::new(false));
        let mut driver =
            TestDriver::new([1; 16], EndpointKind::Npu, DriverContextLifetime::Session);
        driver.advice_enabled = true;
        driver.advice_prefetch_gate = Some(gate.clone());
        let mut session = session(vec![Shared::new(driver).unwrap()]);
        let service = session.memory_advice().unwrap();
        let retained_service = session.memory_advice().unwrap();
        let worker_completed = completed.clone();
        let worker = std::thread::spawn(move || {
            service
                .prefetch(
                    0x10000,
                    4096,
                    AdviceLocation::System,
                    PrefetchCompletion::Migration,
                )
                .unwrap();
            worker_completed.store(true, Ordering::Release);
        });
        gate.0.wait();
        assert!(!completed.load(Ordering::Acquire));
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(session.state().unwrap(), crate::session::SessionState::Open);
        session.begin_retirement().unwrap();
        assert_eq!(
            session.state().unwrap(),
            crate::session::SessionState::Retiring
        );
        assert!(retained_service.set(0x10000, 4096, &[]).is_err());
        assert!(retained_service.query(0x10000, 4096, &[]).is_err());
        assert!(retained_service.capabilities().is_err());
        assert!(session.memory_advice().is_err());
        drop(retained_service);
        gate.1.wait();
        worker.join().unwrap();
        assert!(completed.load(Ordering::Acquire));
        session.destroy().unwrap();
        assert_eq!(
            session.state().unwrap(),
            crate::session::SessionState::Closed
        );
    }
}
