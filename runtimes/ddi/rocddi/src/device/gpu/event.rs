// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Provider-qualified GPU notifications, separate from completion predicates.
//!
//! A wake is a hint to recheck a caller's authoritative predicate. Providers
//! own native notification resources and any internal pages; frontends own
//! callback scheduling and their published signal representation.

#![cfg_attr(
    windows,
    allow(
        unused_variables,
        unused_mut,
        clippy::never_loop,
        reason = "no production notification provider is installed on this target"
    )
)]

use crate::device::Device;
#[cfg(target_os = "linux")]
use crate::driver::kfd_notifications;
use crate::driver::{DeviceDriverState, DriverInstance, DriverInstanceId};
use crate::error::error;
use crate::memory::storage::try_vec_with_capacity;
use crate::session::Session;
use crate::{Error, ErrorKind};

/// Address relationship required by a frontend's signal representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalEventAddressing {
    /// A single address must be usable by both CPU and GPU.
    Identity,
    /// Distinct CPU and GPU mailbox addresses may be published.
    Mapped,
}

impl SignalEventAddressing {
    pub(crate) fn requires_identity(self) -> bool {
        self == Self::Identity
    }
}

/// Mailbox projections qualified while the event and its service remain live.
#[derive(Clone, Copy, Debug)]
pub struct SignalEventMailbox {
    /// CPU address for the host-side signal implementation.
    pub host: usize,
    /// GPU address for a device-side producer.
    pub device: u64,
}

/// Exact device notification protocol, translated by each public ABI adapter.
#[derive(Clone, Copy, Debug)]
pub enum SignalEventProtocol {
    /// KFD signal-page store and interrupt, with its 32-bit event identifier.
    KfdSignal {
        /// Value used for both the mailbox store and KFD interrupt context.
        event_id: u32,
    },
}

enum EventOwner {
    #[cfg(target_os = "linux")]
    Kfd(kfd_notifications::SignalEvent),
    #[cfg(not(target_os = "linux"))]
    Unavailable(std::convert::Infallible),
}

/// Native wake resource retaining its exact provider identity.
pub struct SignalEvent {
    provider: DriverInstanceId,
    owner: EventOwner,
}

impl SignalEvent {
    /// Returns the qualified notification protocol without exposing page state.
    #[must_use]
    pub fn protocol(&self) -> SignalEventProtocol {
        match &self.owner {
            #[cfg(target_os = "linux")]
            EventOwner::Kfd(event) => SignalEventProtocol::KfdSignal {
                event_id: event.info().kfd_event_id,
            },
            #[cfg(not(target_os = "linux"))]
            EventOwner::Unavailable(never) => match *never {},
        }
    }

    /// Signals a native host wake.
    ///
    /// # Errors
    /// Reports provider loss or failure without discarding the owner.
    pub fn signal(&self) -> Result<(), Error> {
        match &self.owner {
            #[cfg(target_os = "linux")]
            EventOwner::Kfd(event) => event.signal(),
            #[cfg(not(target_os = "linux"))]
            EventOwner::Unavailable(never) => match *never {},
        }
    }

    /// Waits for a wake hint, returning false on timeout.
    ///
    /// # Errors
    /// Reports native wait or context failure.
    pub fn wait(&self, timeout_nanoseconds: u64) -> Result<bool, Error> {
        match &self.owner {
            #[cfg(target_os = "linux")]
            EventOwner::Kfd(event) => event.wait(timeout_nanoseconds),
            #[cfg(not(target_os = "linux"))]
            EventOwner::Unavailable(never) => match *never {},
        }
    }

    /// Produces the queue exception descriptor for this native protocol.
    #[must_use]
    pub fn queue_error_event(&self, payload_address: u64) -> super::queue::QueueErrorEvent {
        match &self.owner {
            #[cfg(target_os = "linux")]
            EventOwner::Kfd(event) => event.queue_error_event(payload_address),
            #[cfg(not(target_os = "linux"))]
            EventOwner::Unavailable(never) => match *never {},
        }
    }

    /// Releases the event, retaining retry state after a native failure.
    ///
    /// # Errors
    /// Reports a native release failure; the event remains owned.
    pub fn destroy(&mut self) -> Result<(), Error> {
        match &mut self.owner {
            #[cfg(target_os = "linux")]
            EventOwner::Kfd(event) => event.destroy(),
            #[cfg(not(target_os = "linux"))]
            EventOwner::Unavailable(never) => match *never {},
        }
    }
}

enum ProviderService {
    #[cfg(target_os = "linux")]
    Kfd(kfd_notifications::SignalEventService),
    #[cfg(not(target_os = "linux"))]
    Unavailable(std::convert::Infallible),
}

/// One frontend's services, separately owned for each installed provider.
pub struct SignalEventService {
    process: crate::os::ProcessIdentity,
    providers: Vec<(DriverInstanceId, ProviderService)>,
    unresolved: bool,
}

impl SignalEventService {
    /// Captures the provider catalog without activating a device or native page.
    ///
    /// # Errors
    /// Reports invalid session ownership or metadata allocation failure.
    pub fn new(session: &Session) -> Result<Self, Error> {
        let drivers = session.installed_drivers()?;
        let mut providers = try_vec_with_capacity(drivers.len())?;
        for driver in drivers {
            match driver {
                #[cfg(target_os = "linux")]
                DriverInstance::LinuxKfd(_) => providers.push((
                    driver.instance_id(),
                    ProviderService::Kfd(kfd_notifications::SignalEventService::new(session)),
                )),
                #[cfg(test)]
                DriverInstance::Test(_) => (),
                #[cfg(not(target_os = "linux"))]
                DriverInstance::Unavailable(never) => match *never {},
            }
        }
        Ok(Self {
            process: session.process_identity(),
            providers,
            unresolved: false,
        })
    }

    fn check(&self) -> Result<(), Error> {
        self.process.check().map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "notification service belongs to another process",
            )
        })
    }

    /// Accounts for every provider's internal allocations before session release.
    ///
    /// # Errors
    /// Reports foreign ownership or any live external dependency.
    pub fn preflight_session_destroy(&self, session: &mut Session) -> Result<(), Error> {
        self.check()?;
        let mut allocations = try_vec_with_capacity(self.providers.len())?;
        for (_, provider) in &self.providers {
            match provider {
                #[cfg(target_os = "linux")]
                ProviderService::Kfd(service) => allocations.extend(service.internal_allocation()),
                #[cfg(not(target_os = "linux"))]
                ProviderService::Unavailable(never) => match *never {},
            }
        }
        session.preflight_destroy_with_internal_allocations(&allocations)
    }

    /// Creates a wake resource through the exact owner of an activated device.
    ///
    /// # Errors
    /// Rejects unqualified providers, foreign devices, incompatible addresses or
    /// native failures. Uncertain native page ownership remains with its provider.
    pub fn create(
        &mut self,
        device: &Device,
        addressing: SignalEventAddressing,
    ) -> Result<(SignalEvent, SignalEventMailbox), Error> {
        self.check()?;
        let id = device.endpoint.driver_instance;
        let (_, provider) = self
            .providers
            .iter_mut()
            .find(|(provider, _)| *provider == id)
            .ok_or_else(|| {
                error(
                    ErrorKind::Unsupported,
                    "device has no notification service in this session",
                )
            })?;
        match provider {
            #[cfg(target_os = "linux")]
            ProviderService::Kfd(service) => {
                let (event, mailbox) = service.create(device, addressing)?;
                Ok((
                    SignalEvent {
                        provider: id,
                        owner: EventOwner::Kfd(event),
                    },
                    mailbox,
                ))
            }
            #[cfg(not(target_os = "linux"))]
            ProviderService::Unavailable(never) => match *never {},
        }
    }

    /// Resolves a pooled event's mailbox through its original provider service.
    ///
    /// # Errors
    /// Rejects foreign provider/device combinations and unavailable mappings.
    pub fn mailbox(
        &self,
        device: &Device,
        event: &SignalEvent,
    ) -> Result<SignalEventMailbox, Error> {
        self.check()?;
        if event.provider != device.endpoint.driver_instance {
            return Err(error(
                ErrorKind::InvalidArgument,
                "event and device belong to different providers",
            ));
        }
        let (_, provider) = self
            .providers
            .iter()
            .find(|(id, _)| *id == event.provider)
            .ok_or_else(|| {
                error(
                    ErrorKind::InvalidArgument,
                    "event belongs to another service",
                )
            })?;
        match (provider, &event.owner) {
            #[cfg(target_os = "linux")]
            (ProviderService::Kfd(service), EventOwner::Kfd(event)) => {
                service.mailbox(device, event)
            }
            #[cfg(not(target_os = "linux"))]
            (ProviderService::Unavailable(never), _) => match *never {},
        }
    }

    /// Releases internal service resources before session shutdown.
    ///
    /// Returns whether any provider retired an unconfirmed native resource.
    /// This result remains sticky across retries after another provider fails.
    ///
    /// # Errors
    /// Retains unfinished native resources for a later release attempt.
    pub fn release(&mut self) -> Result<bool, Error> {
        self.check()?;
        for (_, provider) in &mut self.providers {
            match provider {
                #[cfg(target_os = "linux")]
                ProviderService::Kfd(service) => self.unresolved |= service.release()?,
                #[cfg(not(target_os = "linux"))]
                ProviderService::Unavailable(never) => match *never {},
            }
        }
        Ok(self.unresolved)
    }
}

/// Reports current native notification availability for an activated device.
///
/// # Errors
/// Reports provider state failure. This is not an event or capacity reservation.
pub fn signal_event_available(device: &Device) -> Result<bool, Error> {
    match &device.driver_state {
        #[cfg(target_os = "linux")]
        DeviceDriverState::LinuxKfd { .. } => {
            kfd_notifications::process_signal_event_available(device)
        }
        #[cfg(test)]
        DeviceDriverState::Test { .. } => Ok(false),
        #[cfg(not(target_os = "linux"))]
        DeviceDriverState::Unavailable(never) => match *never {},
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::test_driver::TestDriver;
    use crate::memory::device::{DeviceAccess, HostMappingPolicy};
    use crate::memory::host::HostAllocation;
    use crate::memory::storage::Shared;
    use crate::session::DriverContextLifetime;
    use crate::topology::EndpointKind;

    #[test]
    fn internal_allocations_are_counted_per_provider_without_exempting_duplicates() {
        let drivers = [1, 2].map(|id| {
            DriverInstance::Test(
                Shared::new(
                    TestDriver::new([id; 16], EndpointKind::Npu, DriverContextLifetime::Session)
                        .with_host_registration(),
                )
                .unwrap(),
            )
        });
        let mut session =
            Session::from_drivers(Vec::from(drivers), DriverContextLifetime::Session).unwrap();
        let page = crate::memory::host_page_size().unwrap();
        let mut host = HostAllocation::allocate(page, page).unwrap();
        let mut allocations = Vec::new();
        for id in [1, 2] {
            let device = session
                .activate(&session.open_endpoint([id; 16]).unwrap())
                .unwrap();
            allocations.push(
                device
                    .register_host_allocation(
                        &host,
                        &[],
                        HostMappingPolicy::Fine,
                        page,
                        DeviceAccess::READ,
                    )
                    .unwrap(),
            );
        }
        assert_eq!(
            session.preflight_destroy().unwrap_err().kind(),
            ErrorKind::Busy
        );
        assert_eq!(
            session
                .preflight_destroy_with_internal_allocations(&[&allocations[0]])
                .unwrap_err()
                .kind(),
            ErrorKind::Busy
        );
        assert_eq!(
            session
                .preflight_destroy_with_internal_allocations(&[&allocations[0], &allocations[0]])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
        session
            .preflight_destroy_with_internal_allocations(
                &allocations
                    .iter()
                    .map(|allocation| &**allocation)
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        for allocation in &mut allocations {
            allocation.free().unwrap();
        }
        drop(allocations);
        host.free().unwrap();
        session.destroy().unwrap();
    }

    #[test]
    fn absent_notification_capability_creates_no_native_resource() {
        let driver = DriverInstance::Test(
            Shared::new(TestDriver::new(
                [1; 16],
                EndpointKind::Npu,
                DriverContextLifetime::Session,
            ))
            .unwrap(),
        );
        let mut session =
            Session::from_drivers(vec![driver], DriverContextLifetime::Session).unwrap();
        let mut services = SignalEventService::new(&session).unwrap();
        let device = session
            .activate(&session.open_endpoint([1; 16]).unwrap())
            .unwrap();
        assert!(!signal_event_available(&device).unwrap());
        assert!(
            matches!(services.create(&device, SignalEventAddressing::Mapped), Err(error) if error.kind() == ErrorKind::Unsupported)
        );
        drop(device);
        services.preflight_session_destroy(&mut session).unwrap();
        assert!(!services.release().unwrap());
        session.destroy().unwrap();
    }
}
