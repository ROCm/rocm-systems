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
//! contains it.
//!
//! Absent an override, directories are searched in this order (first
//! match wins), as implemented by [`LibSearch::search_dirs`]:
//!
//! 1. Every directory on `$LD_LIBRARY_PATH`.
//! 2. Any backend-specific [`LibSearch::binary_relative_dirs`],
//!    resolved relative to the `mirage` binary. Lets an in-tree
//!    `cargo build` of the monorepo find a freshly-built emulator
//!    without any extra configuration.
//! 3. `$ROCM_HOME` / `$ROCM_PATH` — the ROCm install root
//!    (`<root>/lib`).
//! 4. `../lib` relative to the `mirage` binary.
//! 5. Standard system / ROCm library directories: `/opt/rocm/lib`,
//!    `/usr/local/lib`, `/usr/lib`, `/usr/lib/x86_64-linux-gnu`.

use std::path::{Path, PathBuf};

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
    /// The library file name, e.g. `"libemulator.so"`.
    pub lib_name: &'a str,
    /// Backend-specific directories to probe relative to the `mirage`
    /// binary's own directory (e.g. an in-tree build output). Empty
    /// for backends with no such location.
    pub binary_relative_dirs: &'a [&'a str],
}

impl LibSearch<'_> {
    /// Returns the ordered list of locations searched for this
    /// library, regardless of whether they currently exist. Useful for
    /// user-facing "we looked here" guidance.
    pub fn candidate_paths(&self) -> Vec<PathBuf> {
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
        for dir in self.search_dirs() {
            out.push(dir.join(self.lib_name));
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

        // 1. Every directory on $LD_LIBRARY_PATH.
        if let Some(paths) = non_empty_var("LD_LIBRARY_PATH") {
            for entry in std::env::split_paths(&paths) {
                if !entry.as_os_str().is_empty() {
                    dirs.push(entry);
                }
            }
        }

        // 2. Backend-specific build outputs, relative to the mirage
        //    binary.
        if let Some(dir) = &exe_dir {
            for rel in self.binary_relative_dirs {
                dirs.push(dir.join(rel));
            }
        }

        // 3. ROCm install root ($ROCM_HOME / $ROCM_PATH) lib dir.
        for key in ["ROCM_HOME", "ROCM_PATH"] {
            if let Some(root) = non_empty_var(key) {
                dirs.push(PathBuf::from(root).join("lib"));
            }
        }

        // 4. ../lib relative to the mirage binary.
        if let Some(dir) = &exe_dir {
            dirs.push(dir.join("../lib"));
        }

        // 5. Standard system / ROCm library directories.
        for dir in STANDARD_LIB_DIRS {
            dirs.push(PathBuf::from(dir));
        }

        dirs
    }
}

/// Locate the emulator library described by `search`, returning the
/// first existing path found in priority order (see the module docs).
pub fn find_emulator_lib(search: &LibSearch) -> Option<PathBuf> {
    // Explicit absolute file overrides.
    for key in search.file_env {
        if let Some(p) = non_empty_var(key) {
            let p = PathBuf::from(p);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    // Explicit directory overrides.
    for key in search.dir_env {
        if let Some(p) = non_empty_var(key) {
            let candidate = PathBuf::from(p).join(search.lib_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    // The fixed 5-step directory search.
    for dir in search.search_dirs() {
        let candidate = dir.join(search.lib_name);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Returns `true` if the emulator library can be located on this
/// machine.
pub fn is_lib_installed(search: &LibSearch) -> bool {
    find_emulator_lib(search).is_some()
}

/// Build a multi-line, user-facing guidance string explaining how to
/// install `display_name` so mirage can find `search.lib_name`. Lists
/// the locations that were searched and the most common install
/// options.
pub fn install_guidance(display_name: &str, search: &LibSearch) -> String {
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
    }
    msg
}

fn non_empty_var(key: &str) -> Option<String> {
    match std::env::var(key) {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Convenience helper: a directory contains a usable library.
pub fn dir_has_lib(dir: &Path, lib_name: &str) -> bool {
    dir.join(lib_name).is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn search() -> LibSearch<'static> {
        LibSearch {
            file_env: &["TEST_EMULATOR_LIB"],
            dir_env: &["TEST_EMULATOR_LIB_DIR"],
            lib_name: "libtest-emulator.so",
            binary_relative_dirs: &[],
        }
    }

    #[test]
    fn finds_explicit_file_override() {
        let _g = crate::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("libtest-emulator.so");
        std::fs::write(&lib, b"x").unwrap();
        // SAFETY: guarded by the test env lock.
        unsafe {
            std::env::set_var("TEST_EMULATOR_LIB", &lib);
        }
        let found = find_emulator_lib(&search());
        unsafe {
            std::env::remove_var("TEST_EMULATOR_LIB");
        }
        assert_eq!(found, Some(lib));
    }

    #[test]
    fn finds_via_dir_override() {
        let _g = crate::paths::test_env_lock();
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path().join("libtest-emulator.so");
        std::fs::write(&lib, b"x").unwrap();
        // SAFETY: guarded by the test env lock.
        unsafe {
            std::env::set_var("TEST_EMULATOR_LIB_DIR", tmp.path());
        }
        let found = find_emulator_lib(&search());
        unsafe {
            std::env::remove_var("TEST_EMULATOR_LIB_DIR");
        }
        assert_eq!(found, Some(lib));
    }

    #[test]
    fn missing_lib_yields_none_and_guidance() {
        let _g = crate::paths::test_env_lock();
        // Ensure no override leaks in from the environment.
        unsafe {
            std::env::remove_var("TEST_EMULATOR_LIB");
            std::env::remove_var("TEST_EMULATOR_LIB_DIR");
        }
        let s = LibSearch {
            file_env: &["TEST_EMULATOR_LIB"],
            dir_env: &["TEST_EMULATOR_LIB_DIR"],
            lib_name: "definitely-not-a-real-lib-xyz.so",
            binary_relative_dirs: &[],
        };
        assert!(!is_lib_installed(&s));
        let guidance = install_guidance("TestEmulator", &s);
        assert!(guidance.contains("definitely-not-a-real-lib-xyz.so"));
    }
}
