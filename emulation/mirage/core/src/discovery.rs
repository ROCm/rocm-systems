//! Shared discovery logic for locating emulator runtime libraries.
//!
//! mirage does not bundle or build its emulator backends. Instead it
//! expects the user to install them and then *finds* the relevant
//! shared library at runtime. This module implements the common search
//! policy so every backend looks in the same set of well-known
//! locations.
//!
//! An explicit override always wins first: any env var in
//! [`LibSearch::file_env`] holding an absolute path to the `.so`, then
//! any env var in [`LibSearch::dir_env`] holding a directory that
//! contains it, then any env var in [`LibSearch::home_env`] holding an
//! install root (with the library under `<root>/lib`).
//!
//! Absent an override, the backend-specific
//! [`LibSearch::binary_relative_dirs`] (resolved relative to the
//! `mirage` binary) are always searched — this lets an in-tree
//! `cargo build` of the monorepo find a freshly-built emulator without
//! any extra configuration.
//!
//! Backends that opt in via [`LibSearch::system_fallbacks`] also search
//! a set of generic locations (first match wins), as implemented by
//! `LibSearch::search_dirs`:
//!
//! 1. Every directory on `$LD_LIBRARY_PATH`.
//! 2. `../lib` relative to the `mirage` binary — the library installed
//!    beside it. This one comes *before* the build outputs above: an
//!    installed prefix that happens to sit inside the checkout must
//!    ship its own library rather than whatever the host last built
//!    into `emulation/rocjitsu/build`.
//! 3. `$ROCM_HOME` / `$ROCM_PATH` — the ROCm install root
//!    (`<root>/lib`).
//! 4. The ROCm SDK install root reported by `rocm-sdk path --root`
//!    (`<root>/lib`) — present when a ROCm Python wheel venv is
//!    active.
//! 5. Standard system / ROCm library directories: `/opt/rocm/lib`,
//!    `/usr/local/lib`, `/usr/lib`, `/usr/lib/x86_64-linux-gnu`.
//!
//! Every one of those locations is a fact about the machine mirage is
//! running on, and the only question a user has when a backend reports
//! itself uninstalled is which of them mirage looked at.
//! [`locate_emulator_lib`] therefore returns a [`RuntimeLocation`] — the
//! resolved library, or the list of paths that were probed and the
//! environment variables that would change the answer — and the plain
//! `Option` form ([`find_emulator_lib`]) is a thin wrapper over it, so
//! the search order is defined exactly once.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Standard system / ROCm library directories probed as a last resort.
pub const STANDARD_LIB_DIRS: &[&str] = &[
    "/opt/rocm/lib",
    "/usr/local/lib",
    "/usr/lib",
    "/usr/lib/x86_64-linux-gnu",
];

/// Describes how to locate one emulator's shared library.
#[derive(Debug, Clone)]
pub struct LibSearch<'a> {
    /// Env vars whose value is an absolute path to the `.so` file.
    pub file_env: &'a [&'a str],
    /// Env vars whose value is a directory containing the `.so`.
    pub dir_env: &'a [&'a str],
    /// Env vars whose value is an install *root*, with the library
    /// expected under `<root>/lib` (e.g. `$HOTSWAP_HOME`). Checked
    /// after [`Self::dir_env`] and before the fixed directory search.
    pub home_env: &'a [&'a str],
    /// The library file name, e.g. `"libemulator.so"`.
    pub lib_name: &'a str,
    /// Backend-specific directories to probe relative to the `mirage`
    /// binary's own directory (e.g. an in-tree build output). Empty
    /// for backends with no such location.
    pub binary_relative_dirs: &'a [&'a str],
    /// Whether to also search the generic system fallback locations
    /// (`$LD_LIBRARY_PATH`, `$ROCM_HOME`/`$ROCM_PATH`, `../lib`, and the
    /// standard system/ROCm dirs). Backends with a tightly-scoped
    /// discovery contract (e.g. HotSwap) set this `false` so discovery
    /// is limited to their explicit overrides and build outputs.
    pub system_fallbacks: bool,
}

impl LibSearch<'_> {
    /// Returns the ordered list of locations searched for this
    /// library, regardless of whether they currently exist. Useful for
    /// user-facing "we looked here" guidance.
    pub fn candidate_paths(&self) -> Vec<PathBuf> {
        let mut out = self.override_paths();
        for dir in self.search_dirs() {
            out.push(dir.join(self.lib_name));
        }
        out
    }

    /// The paths named by an explicit override that is actually set:
    /// [`Self::file_env`] verbatim, then [`Self::dir_env`] and
    /// [`Self::home_env`] with the library name appended.
    ///
    /// Split out from [`Self::candidate_paths`] because these are the
    /// only candidates that can be produced without touching the rest
    /// of the machine: assembling the fixed directory list shells out
    /// to `rocm-sdk`, and a user who has already said where the library
    /// is should not pay for that.
    fn override_paths(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        for key in self.file_env {
            if let Some(p) = non_empty_var(key) {
                out.push(PathBuf::from(p));
            }
        }
        for key in self.dir_env {
            if let Some(p) = non_empty_var(key) {
                out.push(PathBuf::from(p).join(self.lib_name));
            }
        }
        for key in self.home_env {
            if let Some(p) = non_empty_var(key) {
                out.push(PathBuf::from(p).join("lib").join(self.lib_name));
            }
        }
        out
    }

    /// The environment variables that would change where this search
    /// looks, in the order they are honoured, each paired with what its
    /// value must point at.
    ///
    /// This is the actionable half of a "not found" report: the list of
    /// probed paths says where mirage looked, and this says what to set
    /// so that it looks somewhere else. It is derived from the same
    /// [`LibSearch`] the search itself uses, so a backend that gains or
    /// loses an override cannot end up advertising the old set.
    pub fn env_hints(&self) -> Vec<EnvHint> {
        let mut out: Vec<EnvHint> = Vec::new();
        for key in self.file_env {
            out.push(EnvHint::new(key, EnvExpects::Library));
        }
        for key in self.dir_env {
            out.push(EnvHint::new(key, EnvExpects::Directory));
        }
        for key in self.home_env {
            out.push(EnvHint::new(key, EnvExpects::Root));
        }
        if self.system_fallbacks {
            out.push(EnvHint::new("LD_LIBRARY_PATH", EnvExpects::Directory));
            out.push(EnvHint::new("ROCM_HOME", EnvExpects::Root));
            out.push(EnvHint::new("ROCM_PATH", EnvExpects::Root));
        }
        out
    }

    /// The fixed directory search order (see the module docs),
    /// excluding the explicit `file_env` / `dir_env` overrides.
    fn search_dirs(&self) -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(Path::to_path_buf));

        // Generic, opt-in: every directory on $LD_LIBRARY_PATH.
        if self.system_fallbacks
            && let Some(paths) = non_empty_var("LD_LIBRARY_PATH")
        {
            for entry in std::env::split_paths(&paths) {
                if !entry.as_os_str().is_empty() {
                    dirs.push(entry);
                }
            }
        }

        if self.system_fallbacks
            && let Some(dir) = &exe_dir
        {
            dirs.push(dir.join("../lib"));
        }

        // Always: backend-specific build outputs, relative to the mirage
        // binary.
        if let Some(dir) = &exe_dir {
            for rel in self.binary_relative_dirs {
                dirs.push(dir.join(rel));
            }
        }

        if self.system_fallbacks {
            // ROCm install root ($ROCM_HOME / $ROCM_PATH) lib dir.
            for key in ["ROCM_HOME", "ROCM_PATH"] {
                if let Some(root) = non_empty_var(key) {
                    dirs.push(PathBuf::from(root).join("lib"));
                }
            }
            // ROCm SDK install root reported by the `rocm-sdk` CLI
            // (present when a ROCm Python wheel venv is active), e.g.
            // `<venv>/lib/pythonX.Y/site-packages/_rocm_sdk_devel/lib`.
            if let Some(root) = rocm_sdk_root() {
                dirs.push(root.join("lib"));
            }
            // Standard system / ROCm library directories.
            for dir in STANDARD_LIB_DIRS {
                dirs.push(PathBuf::from(dir));
            }
        }

        dirs
    }
}

/// What kind of value an environment variable must hold for the search
/// to find the library through it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvExpects {
    /// An absolute path to the library file itself.
    Library,
    /// A directory that contains the library.
    Directory,
    /// An install root, with the library under `<root>/lib`.
    Root,
}

/// One environment variable a user can set to make mirage find a
/// backend's runtime library, and what it expects to be set to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnvHint {
    /// The variable's name, e.g. `ROCJITSU_LIB`.
    pub name: String,
    /// What its value must point at.
    pub expects: EnvExpects,
}

impl EnvHint {
    fn new(name: &str, expects: EnvExpects) -> Self {
        Self {
            name: name.to_string(),
            expects,
        }
    }

    /// A ready-to-paste `NAME=<what to put here>` line naming
    /// `lib_name`, for a user reading a "not found" report.
    #[must_use]
    pub fn assignment(&self, lib_name: &str) -> String {
        let value = match self.expects {
            EnvExpects::Library => format!("<absolute path to {lib_name}>"),
            EnvExpects::Directory => format!("<a directory containing {lib_name}>"),
            EnvExpects::Root => format!("<install root, with lib/{lib_name} under it>"),
        };
        format!("{}={value}", self.name)
    }
}

/// Where a backend's runtime library is on *this* machine — or, when it
/// is not here, where mirage looked for it.
///
/// The second case is the one that matters. "Not installed" on its own
/// leaves the user with no way to act: they cannot tell whether mirage
/// wants the library somewhere they have never heard of, or whether it
/// looked in the one directory they installed it into and rejected it.
/// [`Self::Missing`] therefore carries the probed paths and the
/// environment variables that would change the answer, both taken from
/// the [`LibSearch`] that performed the search rather than re-derived.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum RuntimeLocation {
    /// The library, as resolved on this host.
    Found {
        /// Absolute (or binary-relative) path the search settled on.
        path: PathBuf,
    },
    /// The library was not found anywhere the search looks.
    Missing {
        /// File name that was being looked for, e.g. `librocjitsu.so`.
        lib_name: String,
        /// Every location probed, in the order they were probed.
        searched: Vec<PathBuf>,
        /// Environment variables that would change where mirage looks.
        env: Vec<EnvHint>,
    },
    /// This backend has no runtime library to locate, so there is
    /// nothing to report either way (a backend that is entirely
    /// in-process, or a test stub).
    Unknown,
}

/// How many probed paths the human-readable report shows before it
/// elides the middle of the list. A backend whose search walks every
/// ancestor of the mirage binary for a build tree can probe a hundred
/// locations; printing all of them buries the part of the report the
/// user can act on. The JSON form carries the whole list.
const REPORT_HEAD: usize = 6;
/// How many of the *last* probed paths the report shows after the
/// elision. The tail is where the standard system directories live, so
/// dropping it would hide the locations most users install into.
const REPORT_TAIL: usize = 3;

impl RuntimeLocation {
    /// The library was found at `path`.
    #[must_use]
    pub fn found(path: impl Into<PathBuf>) -> Self {
        Self::Found { path: path.into() }
    }

    /// The resolved library path, if there is one.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        match self {
            Self::Found { path } => Some(path),
            _ => None,
        }
    }

    /// `true` when the library was located on this machine.
    #[must_use]
    pub fn is_found(&self) -> bool {
        matches!(self, Self::Found { .. })
    }

    /// The locations probed, empty unless the library is missing.
    #[must_use]
    pub fn searched(&self) -> &[PathBuf] {
        match self {
            Self::Missing { searched, .. } => searched,
            _ => &[],
        }
    }

    /// The long-form `key: value` lines describing this location, for
    /// `mirage emulators -l`.
    ///
    /// An empty key continues the previous line's value, so the caller
    /// can lay every value out in one column without knowing how many
    /// lines a location needs. [`Self::Unknown`] renders as nothing at
    /// all: a backend with no runtime library has no path to show, and
    /// an empty `runtime:` line would only invite the question.
    ///
    /// The middle of a long `searched` list is elided (see
    /// `REPORT_HEAD`); the serialized form carries every entry, which
    /// is the direction that costs nothing.
    #[must_use]
    pub fn report(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unknown => Vec::new(),
            Self::Found { path } => vec![("runtime", path.display().to_string())],
            Self::Missing {
                lib_name,
                searched,
                env,
            } => {
                let mut out = vec![("runtime", format!("not found ({lib_name})"))];
                let elided = searched.len().saturating_sub(REPORT_HEAD + REPORT_TAIL);
                let shown: Vec<String> = if elided > 1 {
                    searched[..REPORT_HEAD]
                        .iter()
                        .map(|p| p.display().to_string())
                        .chain(std::iter::once(format!(
                            "… and {elided} more (the --json output lists every location)"
                        )))
                        .chain(
                            searched[searched.len() - REPORT_TAIL..]
                                .iter()
                                .map(|p| p.display().to_string()),
                        )
                        .collect()
                } else {
                    searched.iter().map(|p| p.display().to_string()).collect()
                };
                let mut key = "searched";
                for line in shown {
                    out.push((key, line));
                    key = "";
                }
                let mut key = "set";
                for hint in env {
                    out.push((key, hint.assignment(lib_name)));
                    key = "";
                }
                out
            }
        }
    }

    /// One sentence naming what was looked for, how hard, and what to
    /// set — for an error message rather than the `emulators -l` table.
    ///
    /// Returns `None` when there is nothing missing to explain.
    ///
    /// Backends used to write this text out by hand, and both copies had
    /// drifted from the search they described: rocjitsu's listed six of
    /// the nine places it looks, and the DBT translator's named no
    /// environment variable at all. Deriving it from the same
    /// [`RuntimeLocation`] the search produced is what stops that
    /// happening again — a message about a search should come from the
    /// search.
    #[must_use]
    pub fn explain_missing(&self) -> Option<String> {
        let Self::Missing {
            lib_name,
            searched,
            env,
        } = self
        else {
            return None;
        };
        let count = searched.len();
        let places = if count == 1 { "location" } else { "locations" };
        let mut out = format!("searched {count} {places} for {lib_name}");
        if let Some(first) = searched.first() {
            out.push_str(&format!(" (from {}", first.display()));
            if let Some(last) = searched.last().filter(|_| count > 1) {
                out.push_str(&format!(" to {}", last.display()));
            }
            out.push(')');
        }
        if !env.is_empty() {
            let hints: Vec<String> = env.iter().map(|h| h.assignment(lib_name)).collect();
            out.push_str(&format!(". Set one of: {}", hints.join(", ")));
        }
        out.push_str(". `mirage emulators -l` lists every location.");
        Some(out)
    }
}

/// Locate the emulator library described by `search`, reporting either
/// where it is or every location that was probed looking for it.
///
/// This is the one implementation of the search order documented at the
/// top of this module; [`find_emulator_lib`] and [`is_lib_installed`]
/// are views of its result. Callers that only want the path pay nothing
/// extra: the probed paths are accumulated as the search walks them, and
/// the walk still stops at the first hit.
pub fn locate_emulator_lib(search: &LibSearch<'_>) -> RuntimeLocation {
    // Explicit overrides first — an absolute file, a directory, an
    // install root — so a user who has said where the library is gets an
    // answer without mirage assembling (and shelling out for) the rest
    // of the candidate list.
    let mut searched = search.override_paths();
    if let Some(found) = searched.iter().find(|p| p.is_file()) {
        return RuntimeLocation::found(found.clone());
    }
    // The fixed directory search.
    for dir in search.search_dirs() {
        let candidate = dir.join(search.lib_name);
        if candidate.is_file() {
            return RuntimeLocation::found(candidate);
        }
        searched.push(candidate);
    }
    RuntimeLocation::Missing {
        lib_name: search.lib_name.to_string(),
        searched,
        env: search.env_hints(),
    }
}

/// Locate the emulator library described by `search`, returning the
/// first existing path found in priority order (see the module docs).
pub fn find_emulator_lib(search: &LibSearch<'_>) -> Option<PathBuf> {
    match locate_emulator_lib(search) {
        RuntimeLocation::Found { path } => Some(path),
        _ => None,
    }
}

/// Returns `true` if the emulator library can be located on this
/// machine.
pub fn is_lib_installed(search: &LibSearch<'_>) -> bool {
    locate_emulator_lib(search).is_found()
}

/// Build a multi-line, user-facing guidance string explaining how to
/// install `display_name` so mirage can find `search.lib_name`. Lists
/// the locations that were searched and the most common install
/// options.
pub fn install_guidance(display_name: &str, search: &LibSearch<'_>) -> String {
    let mut msg = format!(
        "{display_name} is not installed, or mirage could not find `{lib}`.\n\
         mirage does not build {display_name}; install it yourself and place \
         `{lib}` in any of these locations:\n",
        display_name = display_name,
        lib = search.lib_name,
    );
    for path in search.candidate_paths() {
        msg.push_str(&format!("  - {}\n", path.display()));
    }
    if let Some(file_env) = search.file_env.first() {
        msg.push_str(&format!(
            "\nTip: point `{file_env}` at the library directly, e.g.\n  \
             export {file_env}=/abs/path/to/{lib}\n",
            file_env = file_env,
            lib = search.lib_name,
        ));
    } else if let Some(home_env) = search.home_env.first() {
        msg.push_str(&format!(
            "\nTip: point `{home_env}` at the install root (with `{lib}` \
             under `<root>/lib`), e.g.\n  \
             export {home_env}=/abs/path/to/install-root\n",
            home_env = home_env,
            lib = search.lib_name,
        ));
    }
    msg
}

fn non_empty_var(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Best-effort query of the ROCm SDK install root via the `rocm-sdk`
/// CLI, which is on `PATH` when a ROCm Python wheel venv is active.
/// Returns the trimmed `<root>` from `rocm-sdk path --root`, or `None`
/// if the CLI is unavailable, fails, or prints nothing.
pub fn rocm_sdk_root() -> Option<PathBuf> {
    let out = std::process::Command::new("rocm-sdk")
        .args(["path", "--root"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let root = String::from_utf8(out.stdout).ok()?;
    let root = root.trim();
    if root.is_empty() {
        return None;
    }
    Some(PathBuf::from(root))
}

/// Convenience helper: a directory contains a usable library.
pub fn dir_has_lib(dir: &Path, lib_name: &str) -> bool {
    dir.join(lib_name).is_file()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn missing_lib_yields_none_and_guidance() {
        // Reference env var names that are never set so the lookup falls
        // through to "not found" without mutating the environment.
        let s = LibSearch {
            file_env: &["MIRAGE_NONEXISTENT_TEST_EMULATOR_LIB"],
            dir_env: &["MIRAGE_NONEXISTENT_TEST_EMULATOR_LIB_DIR"],
            home_env: &[],
            lib_name: "definitely-not-a-real-lib-xyz.so",
            binary_relative_dirs: &[],
            system_fallbacks: true,
        };
        assert!(!is_lib_installed(&s));
        let guidance = install_guidance("TestEmulator", &s);
        assert!(guidance.contains("definitely-not-a-real-lib-xyz.so"));
    }

    /// The library installed beside the binary outranks any in-tree
    /// build directory.
    #[test]
    fn the_installed_sibling_outranks_an_in_tree_build() {
        let s = LibSearch {
            file_env: &[],
            dir_env: &[],
            home_env: &[],
            lib_name: "definitely-not-a-real-lib-xyz.so",
            binary_relative_dirs: &["rocjitsu/build"],
            system_fallbacks: true,
        };

        let candidates = s.candidate_paths();
        let position = |needle: &str| {
            candidates
                .iter()
                .position(|p| p.to_string_lossy().contains(needle))
                .unwrap_or_else(|| panic!("{needle} is not searched at all: {candidates:?}"))
        };

        assert!(
            position("../lib") < position("rocjitsu/build"),
            "the installed sibling must be searched first: {candidates:?}"
        );
    }

    /// A backend with a tightly-scoped discovery contract keeps it: the
    /// installed-sibling rule is part of the generic fallbacks, so it
    /// must stay behind `system_fallbacks` where it was moved from.
    #[test]
    fn the_installed_sibling_is_a_system_fallback() {
        let s = LibSearch {
            file_env: &[],
            dir_env: &[],
            home_env: &[],
            lib_name: "definitely-not-a-real-lib-xyz.so",
            binary_relative_dirs: &["rocjitsu/build"],
            system_fallbacks: false,
        };

        assert!(
            !s.candidate_paths()
                .iter()
                .any(|p| p.to_string_lossy().contains("../lib")),
            "an opted-out backend must not gain `../lib`"
        );
    }

    /// A search that finds nothing must say where it looked and what to
    /// set, because "not found" on its own is the one answer a user
    /// cannot act on.
    #[test]
    fn a_missing_lib_reports_where_it_looked_and_what_to_set() {
        let s = LibSearch {
            file_env: &["MIRAGE_NONEXISTENT_TEST_EMULATOR_LIB"],
            dir_env: &[],
            home_env: &[],
            lib_name: "definitely-not-a-real-lib-xyz.so",
            binary_relative_dirs: &["../nowhere"],
            system_fallbacks: true,
        };

        let location = locate_emulator_lib(&s);

        let RuntimeLocation::Missing {
            lib_name,
            searched,
            env,
        } = &location
        else {
            panic!("a library that does not exist cannot be found: {location:?}");
        };
        assert_eq!(lib_name, s.lib_name);
        assert!(!searched.is_empty(), "the probed locations must be named");
        // Every probed path is a full path to the library, and the
        // standard directories are among them.
        assert!(searched.iter().all(|p| p.ends_with(s.lib_name)));
        assert!(searched.contains(&PathBuf::from("/opt/rocm/lib").join(s.lib_name)));
        // And the way out is named: the backend's own override first,
        // then the generic ROCm locations it opted into.
        let names: Vec<&str> = env.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(names.first(), Some(&"MIRAGE_NONEXISTENT_TEST_EMULATOR_LIB"));
        assert!(names.contains(&"LD_LIBRARY_PATH"));
        assert!(names.contains(&"ROCM_PATH"));
        assert!(env[0].assignment(s.lib_name).contains(s.lib_name));
    }

    /// The `Option` form and the reporting form are the same search:
    /// whatever one says about this host, the other must agree with.
    #[test]
    fn find_and_locate_agree() {
        let s = LibSearch {
            file_env: &[],
            dir_env: &[],
            home_env: &[],
            lib_name: "definitely-not-a-real-lib-xyz.so",
            binary_relative_dirs: &[],
            system_fallbacks: false,
        };
        assert_eq!(
            find_emulator_lib(&s).as_deref(),
            locate_emulator_lib(&s).path()
        );
        assert_eq!(find_emulator_lib(&s).is_some(), is_lib_installed(&s));
    }

    /// A long probe list is elided in the middle rather than at the
    /// end: the first entries are the build outputs beside the binary
    /// and the last are the standard system directories, and a user
    /// needs to see both.
    #[test]
    fn a_long_report_keeps_both_ends_of_the_search() {
        let searched: Vec<PathBuf> = (0..40)
            .map(|i| PathBuf::from(format!("/d{i}/lib.so")))
            .collect();
        let location = RuntimeLocation::Missing {
            lib_name: "lib.so".to_string(),
            searched,
            env: vec![EnvHint::new("SOME_LIB", EnvExpects::Library)],
        };

        let report = location.report();
        let text = report
            .iter()
            .map(|(_, value)| value.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains("/d0/lib.so"), "{text}");
        assert!(text.contains("/d39/lib.so"), "{text}");
        assert!(text.contains("31 more"), "{text}");
        assert!(
            text.contains("SOME_LIB=<absolute path to lib.so>"),
            "{text}"
        );
        // Keys label the first line of each group and are blank on its
        // continuations, so the caller can align one column.
        let keys: Vec<&str> = report.iter().map(|(key, _)| *key).collect();
        assert_eq!(keys[0], "runtime");
        assert_eq!(keys[1], "searched");
        assert_eq!(keys[2], "");
        assert_eq!(keys.iter().filter(|k| **k == "set").count(), 1);
    }

    /// A backend with no runtime library to find reports nothing rather
    /// than an empty `runtime:` line.
    #[test]
    fn an_unknown_location_reports_nothing() {
        assert!(RuntimeLocation::Unknown.report().is_empty());
        assert!(!RuntimeLocation::Unknown.is_found());
        assert!(RuntimeLocation::Unknown.searched().is_empty());
    }
}
