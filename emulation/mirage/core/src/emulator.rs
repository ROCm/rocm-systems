//! Emulator backends: the trait every GPU emulator integration
//! implements, and the link-time registry they are discovered through.

use serde::{Deserialize, Serialize};

use crate::{
    common::{MaybeRef, SimpleMap},
    config::OptionDef,
    discovery::RuntimeLocation,
    error::Result,
    exec::InjectionDef,
    plugin::PluginsDef,
    profile::ProfileDef,
    session::{SessionContext, SessionHealth},
    topology::TopologyDef,
};

/// How faithfully the emulator models the device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ExecMode {
    /// Correct results, no timing model.
    #[default]
    Functional,
    /// Model device timing as well as results.
    Clocked,
}

/// The canonical lowercase name of an emulator backend.
pub type EmulatorKind = String;

/// The emulator half of a profile: which backend, how it runs, and the
/// system it emulates.
///
/// Unknown fields are rejected; see [`crate::profile`] for why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmulatorDef {
    /// Which emulator backend runs this profile, by its canonical
    /// lowercase name.
    pub emulator: EmulatorKind,

    /// Which of the backend's plugins to enable, each with its argument
    /// object. See [`PluginsDef`] for what an empty object means.
    pub plugins: PluginsDef,

    /// Functional or clocked emulation.
    pub exec_mode: ExecMode,

    /// Backend-specific configuration overrides, e.g.
    /// `{"gpu_model": "cdna3"}`.
    pub options: SimpleMap,

    /// System topology (rack/node/GPU layout plus the per-GPU agent).
    pub topology: MaybeRef<TopologyDef>,
}

/// Whether the host's hardware/environment can actually run an
/// emulator. This is distinct from [`EmulatorBackend::installed`]:
/// an emulator can be installed yet unsupported (e.g. HotSwap installed
/// on a machine with no compatible physical GPU), or supported yet not
/// installed. Both signals are surfaced so the UX/CLI can explain
/// exactly what a user needs to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportStatus {
    /// `true` if this host meets the emulator's hardware/environment
    /// requirements.
    pub supported: bool,
    /// Human-readable explanation of the support decision — what was
    /// required and what was found. Always populated so the UX/CLI can
    /// show a reason whether or not the host is supported.
    pub reason: String,
}

impl SupportStatus {
    /// The host meets this emulator's requirements.
    pub fn supported(reason: impl Into<String>) -> Self {
        Self {
            supported: true,
            reason: reason.into(),
        }
    }

    /// The host does not meet this emulator's requirements.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        Self {
            supported: false,
            reason: reason.into(),
        }
    }
}

/// The live state of a backend's runtime on this host: whether it is
/// usable, and where it is — or, when it is not here, where mirage
/// looked for it.
///
/// The two travel together because they come from one search. Locating a
/// backend's library means stat'ing every candidate the discovery policy
/// names, which for a backend that hunts for a build tree beside the
/// binary is a hundred paths; asking "installed?" and "where?" as two
/// questions would do that walk twice on the way to printing one line.
///
/// `installed` is not simply "the library exists": a backend may need
/// more than one artifact co-located (HotSwap needs its intercept, a
/// patched ROCR and COMGR in one directory), so it is reported
/// separately rather than derived from `location`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStatus {
    /// `true` if this backend's runtime is present and usable here.
    pub installed: bool,
    /// Where the runtime library is, or where mirage looked.
    pub location: RuntimeLocation,
}

impl RuntimeStatus {
    /// A status for a backend whose install *is* just "the library was
    /// located": installed exactly when the search found it.
    #[must_use]
    pub fn from_location(location: RuntimeLocation) -> Self {
        Self {
            installed: location.is_found(),
            location,
        }
    }

    /// A status for a backend that needs more than a located library to
    /// count as installed, with `installed` decided by the backend.
    #[must_use]
    pub fn new(installed: bool, location: RuntimeLocation) -> Self {
        Self {
            installed,
            location,
        }
    }
}

/// The static identity of an emulator backend: its name, version, a
/// short human-readable blurb, and the schema of options it accepts.
/// Live runtime status (installed / supported) is reported separately
/// through the [`EmulatorBackend`] trait methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmulatorDescription {
    pub name: String,
    pub version: String,
    pub description: String,
    /// Schema of the options this backend accepts (empty when none).
    pub options_schema: Vec<OptionDef>,
}

/// A backend integration mirage can drive to emulate a workload.
///
/// Every emulator (`rocjitsu`, `rocjitsu-dbt`, `hotswap`, …)
/// lives in its own crate and registers a single stateless
/// implementation of this trait into the global registry via
/// [`inventory`]. The core and control-plane crates never name a
/// concrete backend: they look one up by its [`EmulatorKind`] with
/// [`get_emulator_backend`] and dispatch through this trait, so all
/// backend-specific behaviour stays inside the backend's own crate.
///
/// Implementations are stateless singletons (a unit struct is typical);
/// any per-session state is resolved on demand from the on-disk session
/// definition (see `mirage_supervisor::session::resolve_profile`).
pub trait EmulatorBackend: Sync + Send + std::fmt::Debug {
    /// Returns a description of the emulator, including its name, version, and a brief description.
    fn description(&self) -> EmulatorDescription;

    /// Bring the emulator up for the given profile.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason on failure.
    fn boot(&self, def: &ProfileDef) -> std::result::Result<(), String>;

    /// Schema for the options that this emulator supports. Empty when
    /// the emulator takes no options.
    fn options(&self) -> Vec<OptionDef>;

    /// Shut the emulator down for a session and release any resources it
    /// holds. Called exactly once, during session teardown, after every
    /// workload process is gone.
    fn shutdown(&self, ctx: &SessionContext);

    /// Validate that `def` can be used with this emulator before it is
    /// persisted.
    ///
    /// # Errors
    ///
    /// Returns a human-readable reason on rejection.
    fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String>;

    /// Whether this backend's runtime is installed *and* where it was
    /// found, or the locations that were searched when it was not.
    ///
    /// This is what [`crate::registry::registry`] probes with, and the
    /// two halves come from one search on purpose: locating a backend's
    /// library means stat'ing every candidate its discovery policy names,
    /// and asking "installed?" and "where?" separately would walk that
    /// list twice to print one line.
    ///
    /// Required rather than defaulted, and [`Self::installed`] defaulted
    /// from it rather than the other way round, because the location is
    /// the half a user can act on. It was the other way round once: the
    /// compiler asked a new backend author for the bare "no" and let them
    /// skip the actionable answer, whose absence renders as nothing at
    /// all in `mirage emulators -l`.
    ///
    /// A backend with no runtime library to locate has nothing to report
    /// either way and should return
    /// `RuntimeStatus::new(<installed>, RuntimeLocation::Unknown)`; one
    /// that discovers a library should return the [`RuntimeLocation`] its
    /// search produced (see [`crate::discovery::locate_emulator_lib`]).
    fn runtime(&self) -> RuntimeStatus;

    /// Returns true if the emulator is properly installed and can be
    /// used.
    ///
    /// Defaults to [`Self::runtime`]'s verdict, which is where the
    /// question is actually answered; a backend has no reason to
    /// override this and every reason not to, since two implementations
    /// of "is it installed?" can disagree.
    fn installed(&self) -> bool {
        self.runtime().installed
    }

    /// check if the emulator is supported on this host, i.e. meets the hardware/environment requirements to run. This is a stronger condition than `installed`: an emulator can be installed but unsupported (e.g. HotSwap installed on a machine with no compatible physical GPU), or supported but not installed.
    fn supported(&self) -> SupportStatus;

    /// Discovers available plugins for the emulator.
    fn discover_plugins(&self) -> Vec<PluginsDef>;

    /// Health of the emulator for a session, e.g. whether the underlying
    /// runtime is present and responsive.
    fn health(&self, ctx: &SessionContext) -> SessionHealth;

    /// Compute the env vars / `LD_PRELOAD` / files to inject into a
    /// workload run under this emulator. Returns an error when the
    /// emulator is selected but its runtime library or assets are
    /// missing, so a misconfigured session fails loudly instead of
    /// silently running unemulated.
    ///
    /// `ctx` carries the session's resolved profile and a scratch
    /// directory the backend may materialise runtime assets in.
    ///
    /// # Errors
    ///
    /// Returns an error when the backend is selected but cannot produce a
    /// usable injection — a missing runtime library, an unresolvable
    /// topology or agent reference, an unwritable scratch directory. This
    /// must fail loudly: silently returning an empty injection would run
    /// the workload unemulated on whatever hardware is actually present.
    fn injection_def(&self, ctx: &SessionContext) -> Result<InjectionDef>;

    /// Start a host-side emulator *daemon* for `session`, if this
    /// backend hosts one.
    ///
    /// The supervisor calls this once during session bring-up, before
    /// any exec runs. A backend that emulates the GPU out-of-process
    /// (e.g. rocjitsu's daemon, which owns the simulated device and
    /// serves the workload's KFD ioctls over a Unix socket) stands the
    /// daemon up here and returns a handle the supervisor keeps alive for
    /// the whole session, stopping it during teardown after every
    /// workload process has exited.
    ///
    /// Returns `Ok(None)` — the default — for backends that need no
    /// daemon (`hotswap`, or rocjitsu when its runtime library
    /// is not installed and the exec will fail loudly anyway).
    ///
    /// # Errors
    ///
    /// Returns `Err` only when a daemon was expected but could not be
    /// started.
    fn start_daemon(&self, ctx: &SessionContext) -> Result<Option<Box<dyn EmulatorDaemon>>> {
        let _ = ctx;
        Ok(None)
    }

    /// Whether this backend could host a daemon here, asked without
    /// creating anything.
    ///
    /// [`Self::start_daemon`] is the last step of bring-up, and for a
    /// containerised session everything expensive has happened by the
    /// time it runs: the image is pulled, the network is up, every node
    /// container is created. A backend whose runtime simply cannot host a
    /// daemon — an installed library that predates the daemon API, say —
    /// fails all of that at the very end, on a fact that was knowable
    /// before any of it started. The supervisor asks this first, so the
    /// same session fails in a second with the same message.
    ///
    /// This must be cheap and side-effect-free enough to ask on every
    /// bring-up and from [`Self::health`]: it decides nothing a backend
    /// has to *do*, only what it would find if it tried. Backends that
    /// host no daemon need not implement it — the default agrees with
    /// [`Self::start_daemon`]'s, which is that there is nothing to host
    /// and therefore nothing that can be missing.
    ///
    /// # Errors
    ///
    /// A human-readable reason the daemon could not be started here,
    /// phrased for a user who has just been refused a run.
    fn daemon_capability(&self) -> Result<()> {
        Ok(())
    }
}

/// A running, host-side emulator daemon owned by a per-node host.
///
/// The host holds the boxed handle for the lifetime of the session and
/// drops it on shutdown. Implementations must tear the daemon down in
/// their [`Drop`] so cleanup happens even if the host panics; [`stop`]
/// is provided for an explicit, ordered shutdown and defaults to simply
/// dropping the handle.
///
/// [`stop`]: EmulatorDaemon::stop
pub trait EmulatorDaemon: Send + std::fmt::Debug {
    /// Stop the daemon and release its resources. Blocking, so the
    /// supervisor calls it from a blocking task rather than inline on the
    /// runtime. The default drops the handle, which must perform the
    /// teardown.
    fn stop(self: Box<Self>) {}
}

/// One registry entry: the canonical [`EmulatorKind`] name plus the
/// backend that handles it. Each backend crate submits exactly one of
/// these via [`inventory::submit!`]; the backend is a `'static`
/// reference to a stateless singleton (typically a unit struct), so the
/// entry is const-constructible and needs no allocation.
#[derive(Debug)]
pub struct EmulatorBackendDef {
    /// Canonical lowercase name the backend registers under (the value
    /// stored in [`EmulatorDef::emulator`]).
    pub kind: &'static str,
    /// The backend implementation.
    pub backend: &'static dyn EmulatorBackend,
}

inventory::collect!(EmulatorBackendDef);

/// Look up the [`EmulatorBackend`] registered for `kind`, or `None` if
/// no backend with that name was compiled into this binary (e.g. its
/// crate's feature is disabled).
#[must_use]
pub fn get_emulator_backend(kind: &str) -> Option<&'static dyn EmulatorBackend> {
    inventory::iter::<EmulatorBackendDef>
        .into_iter()
        .find(|def| def.kind == kind)
        .map(|def| def.backend)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// A backend for this crate's own tests.
    ///
    /// No backend crate can be linked here — they all depend on
    /// `mirage_core` — so anything that resolves an emulator, such as
    /// [`ProfileDef::validate`] on the way into the store, would find an
    /// empty registry. This is the one entry those tests write profiles
    /// against, and it accepts everything: what they exercise is the
    /// store's own rules, not a backend's.
    #[derive(Debug)]
    pub(super) struct Accepting;

    /// The name a test profile puts in its `emulator` field.
    pub(super) const TEST_EMULATOR: &str = "test";

    impl EmulatorBackend for Accepting {
        fn description(&self) -> EmulatorDescription {
            EmulatorDescription {
                name: TEST_EMULATOR.to_string(),
                version: "0".to_string(),
                description: "test-only backend that accepts every profile".to_string(),
                options_schema: Vec::new(),
            }
        }

        fn boot(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
            Ok(())
        }

        fn options(&self) -> Vec<OptionDef> {
            Vec::new()
        }

        fn shutdown(&self, _ctx: &SessionContext) {}

        fn validate_profile(&self, _def: &ProfileDef) -> std::result::Result<(), String> {
            Ok(())
        }

        fn runtime(&self) -> RuntimeStatus {
            // Nothing to locate: this backend is compiled in rather than
            // installed, so there is no library and no search to report.
            RuntimeStatus::new(true, RuntimeLocation::Unknown)
        }

        fn supported(&self) -> SupportStatus {
            SupportStatus::supported("the test backend needs nothing")
        }

        fn discover_plugins(&self) -> Vec<PluginsDef> {
            Vec::new()
        }

        fn health(&self, _ctx: &SessionContext) -> SessionHealth {
            SessionHealth::phase(true, crate::session::state::READY, None)
        }

        fn injection_def(&self, _ctx: &SessionContext) -> Result<InjectionDef> {
            Ok(InjectionDef::default())
        }
    }

    inventory::submit! {
        EmulatorBackendDef { kind: TEST_EMULATOR, backend: &Accepting }
    }

    /// A backend that answers only the question with an answer in it.
    ///
    /// It is [`Accepting`] with [`EmulatorBackend::installed`] left to
    /// the trait, which is the shape the inverted pair is meant to
    /// produce: the compiler demands the location, and the bare yes/no
    /// falls out of it. It is not registered — nothing looks it up — so
    /// it exists purely to hold that shape to the fire at compile time.
    #[derive(Debug)]
    struct LocationOnly;

    /// A library name no machine has, so the search below always misses.
    const NO_SUCH_LIB: &str = "libmirage-emulator-trait-test.so";

    impl EmulatorBackend for LocationOnly {
        fn description(&self) -> EmulatorDescription {
            Accepting.description()
        }

        fn boot(&self, def: &ProfileDef) -> std::result::Result<(), String> {
            Accepting.boot(def)
        }

        fn options(&self) -> Vec<OptionDef> {
            Accepting.options()
        }

        fn shutdown(&self, ctx: &SessionContext) {
            Accepting.shutdown(ctx);
        }

        fn validate_profile(&self, def: &ProfileDef) -> std::result::Result<(), String> {
            Accepting.validate_profile(def)
        }

        fn runtime(&self) -> RuntimeStatus {
            RuntimeStatus::from_location(RuntimeLocation::Missing {
                lib_name: NO_SUCH_LIB.to_string(),
                searched: vec![std::path::PathBuf::from("/nowhere").join(NO_SUCH_LIB)],
                env: Vec::new(),
            })
        }

        fn supported(&self) -> SupportStatus {
            Accepting.supported()
        }

        fn discover_plugins(&self) -> Vec<PluginsDef> {
            Accepting.discover_plugins()
        }

        fn health(&self, ctx: &SessionContext) -> SessionHealth {
            Accepting.health(ctx)
        }

        fn injection_def(&self, ctx: &SessionContext) -> Result<InjectionDef> {
            Accepting.injection_def(ctx)
        }
    }

    /// The pair used to run the other way round: `installed` was
    /// required and `runtime` defaulted to
    /// `RuntimeStatus::new(self.installed(), RuntimeLocation::Unknown)`,
    /// which renders as nothing — so a backend author was asked for the
    /// bare "no" and allowed to skip the answer `mirage emulators -l`
    /// exists to give. Now the location is what a backend must supply
    /// and the flag is derived from it, so the two cannot disagree
    /// either.
    #[test]
    fn installed_comes_from_the_runtime_search() {
        let backend = LocationOnly;
        assert!(!backend.installed());
        assert_eq!(backend.installed(), backend.runtime().installed);
        // And the answer a user can act on came with it.
        assert_eq!(backend.runtime().location.searched().len(), 1);
    }

    #[test]
    fn an_unregistered_backend_is_not_found() {
        assert!(get_emulator_backend(TEST_EMULATOR).is_some());
        assert!(get_emulator_backend("no-such-backend").is_none());
    }
}
