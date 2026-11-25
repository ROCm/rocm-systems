// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Owned Linux KFD options installed before GPU activation.
//!
//! Frontends assemble a complete configuration. The driver stores the options
//! passively until activation applies them to its native context.

use super::LinuxKfdDriver;
use crate::driver::{ConfigurableDriver, DriverInstance};
use crate::session::Session;
use crate::{Error, ErrorKind};
use std::sync::atomic::Ordering;

/// Native recoverable-page-fault mode requested before GPU activation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum XnackModeRequest {
    /// Activation must confirm this mode or report a native configuration error.
    Require(bool),
    /// Attempt this mode; a confirmed existing mode is an acceptable fallback.
    /// A failed live-mode query still prevents activation.
    Prefer(bool),
}

/// Whether activation requires ownership of the native debugger rendezvous.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DebuggerRegistrationMode {
    /// Reject an active context registered to another adapter.
    Require,
    /// Preserve an active context and report unavailable debugger ownership.
    Prefer,
}

/// Debugger registration descriptor for native runtime enablement.
///
/// This value owns the registration options, not the native record or its
/// backing memory. Its construction contract permits the driver and native
/// context to retain copies after the configuring session has ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DebuggerRegistration {
    /// Stable native `r_debug` address retained by the registering adapter.
    pub(super) address: u64,
    /// Requests trap-temporary save/setup as part of runtime enablement.
    pub(super) setup_ttmp: bool,
    /// Whether an already-active different adapter may keep its registration.
    pub(super) mode: DebuggerRegistrationMode,
}

impl DebuggerRegistration {
    /// Describes a native debugger rendezvous without publishing it to KFD.
    ///
    /// # Safety
    /// For a non-null, aligned address, `r_debug` must address a correctly
    /// initialized native `r_debug` record. The record must remain valid for
    /// the process lifetime; its breakpoint and every reachable link-map record
    /// must remain valid while published. Process-scoped contexts can outlive
    /// the configuring session and share this registration.
    /// `setup_ttmp` must match the adapter's debugger/trap-handler ABI.
    ///
    /// # Errors
    /// Rejects null or misaligned addresses without accessing the record.
    #[allow(unsafe_code)]
    pub unsafe fn new(
        r_debug: usize,
        setup_ttmp: bool,
        mode: DebuggerRegistrationMode,
    ) -> Result<Self, Error> {
        if r_debug == 0 || r_debug % 8 != 0 {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid runtime debugger record address",
            });
        }
        Ok(Self {
            address: r_debug as u64,
            setup_ttmp,
            mode,
        })
    }
}

/// Complete native activation options owned by one Linux KFD driver.
///
/// The default requests neither an XNACK mode nor debugger registration.
/// These are requested settings; activation reconciles them with an existing
/// shared native context, whose effective settings can differ when preferred.
/// Pass the complete value to [`Session::configure`] before activation. Absent
/// options remove prior pending requests; they do not clear settings or
/// unregister a debugger in an active shared native context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KfdConfiguration {
    /// Required or preferred recoverable-page-fault mode; `None` makes no request.
    pub xnack: Option<XnackModeRequest>,
    /// Debugger rendezvous and policy; `None` makes no registration request.
    pub debugger: Option<DebuggerRegistration>,
}

impl crate::driver::configuration::Apply for KfdConfiguration {
    fn apply(self, session: &mut Session) -> Result<(), Error> {
        session.configure_driver(self, |driver| match driver {
            DriverInstance::LinuxKfd(driver) => Some(driver),
            #[cfg(test)]
            DriverInstance::Test(_) => None,
        })
    }
}

impl ConfigurableDriver for LinuxKfdDriver {
    type Configuration = KfdConfiguration;

    fn configure(&mut self, configuration: Self::Configuration) -> Result<(), Error> {
        self.ensure_open()?;
        if self.joined_primary.load(Ordering::Acquire) || self.local.kfd.get().is_some() {
            return Err(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "KFD configuration must precede GPU activation",
            });
        }
        self.configuration = configuration;
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::test_driver::TestDriver;
    use crate::memory::storage::Shared;
    use crate::os::ProcessIdentity;
    use crate::session::DriverContextLifetime;
    use crate::topology::{EndpointKind, GpuInfo};

    fn requested_configuration() -> KfdConfiguration {
        KfdConfiguration {
            xnack: Some(XnackModeRequest::Require(true)),
            // This backend-only fixture never publishes the synthetic address.
            debugger: Some(DebuggerRegistration {
                address: 0x1000,
                setup_ttmp: true,
                mode: DebuggerRegistrationMode::Prefer,
            }),
        }
    }

    #[test]
    fn configuration_checks_session_and_driver_access() {
        let mut session = Session::new(DriverContextLifetime::Session).unwrap();
        let shared = session.clone();
        assert_eq!(
            session
                .configure(requested_configuration())
                .unwrap_err()
                .kind(),
            ErrorKind::Busy
        );
        drop(shared);

        let retained_driver = session.installed_drivers().unwrap()[0]
            .linux_kfd()
            .unwrap()
            .clone();
        assert_eq!(
            session
                .configure(requested_configuration())
                .unwrap_err()
                .kind(),
            ErrorKind::Busy
        );
        assert_eq!(retained_driver.configuration, KfdConfiguration::default());
        drop(retained_driver);
        session.configure(requested_configuration()).unwrap();
        session.destroy().unwrap();
        assert_eq!(
            session
                .configure(requested_configuration())
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
    }

    #[test]
    fn configuration_replaces_all_options_passively_before_activation() {
        let mut session = Session::new(DriverContextLifetime::Session).unwrap();
        let requested = requested_configuration();
        session.configure(requested).unwrap();
        let driver = session.installed_drivers().unwrap()[0].linux_kfd().unwrap();
        assert_eq!(driver.configuration, requested);
        assert!(driver.local.kfd.get().is_none());

        session.configure(KfdConfiguration::default()).unwrap();
        let driver = session.installed_drivers().unwrap()[0].linux_kfd().unwrap();
        assert_eq!(driver.configuration, KfdConfiguration::default());
        assert!(driver.local.kfd.get().is_none());
        driver.joined_primary.store(true, Ordering::Release);
        assert_eq!(
            session.configure(requested).unwrap_err().kind(),
            ErrorKind::Busy
        );
        let driver = session.installed_drivers().unwrap()[0].linux_kfd().unwrap();
        assert_eq!(driver.configuration, KfdConfiguration::default());
        driver.joined_primary.store(false, Ordering::Release);
        session.destroy().unwrap();
    }

    #[test]
    fn configuration_rejects_other_drivers() {
        let id = [0x24; 16];
        let driver = TestDriver::new(
            id,
            EndpointKind::Gpu {
                info: GpuInfo::default(),
            },
            DriverContextLifetime::Session,
        );
        let mut session = Session::from_drivers(
            vec![DriverInstance::Test(Shared::new(driver).unwrap())],
            DriverContextLifetime::Session,
        )
        .unwrap();
        assert_eq!(
            session
                .configure(requested_configuration())
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
        session.destroy().unwrap();
    }

    #[test]
    fn configuration_is_fixed_after_a_local_context_is_opened() {
        let mut driver = LinuxKfdDriver::new();
        driver.configure(requested_configuration()).unwrap();
        let kfd = super::super::connection::Kfd::new(
            std::fs::File::open("/dev/null").unwrap(),
            ProcessIdentity::capture().unwrap(),
        );
        assert!(driver.local.kfd.set(Shared::new(kfd).unwrap()).is_ok());
        assert_eq!(
            driver
                .configure(KfdConfiguration::default())
                .unwrap_err()
                .kind(),
            ErrorKind::Busy
        );
        assert_eq!(driver.configuration, requested_configuration());
    }

    #[test]
    fn configuration_rejects_inherited_and_closing_drivers_before_mutation() {
        let mut driver = LinuxKfdDriver::new();
        driver.configure(requested_configuration()).unwrap();
        driver.process = ProcessIdentity::with_pid_for_test(std::process::id().wrapping_add(1));
        assert_eq!(
            driver
                .configure(KfdConfiguration::default())
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(driver.configuration, requested_configuration());
        driver.process = ProcessIdentity::capture().unwrap();
        driver.closing = true;
        assert_eq!(
            driver
                .configure(KfdConfiguration::default())
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(driver.configuration, requested_configuration());
    }

    #[test]
    #[allow(unsafe_code)]
    fn debugger_registration_rejects_invalid_addresses() {
        for address in [0, 4] {
            // SAFETY: Null and misaligned addresses are rejected without access.
            let result = unsafe {
                DebuggerRegistration::new(address, true, DebuggerRegistrationMode::Require)
            };
            assert_eq!(result.unwrap_err().kind(), ErrorKind::InvalidArgument);
        }
    }
}
