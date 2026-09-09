//! The ROCm runtime a workload will load, and the GPU targets it knows
//! how to enumerate.
//!
//! An emulated device is only half of a working session. mirage (through
//! rocjitsu) synthesises a KFD node for the GPU a profile describes, and
//! then something has to *recognise* it: `libhsa-runtime64.so`, the ROCm
//! runtime the workload itself loads, reads that node's
//! `gfx_target_version`, looks the ISA up in the table it was compiled
//! with, and — when the ISA is not in that table — skips the agent
//! entirely.
//!
//! Skips it silently. The emulated device is presented correctly, the
//! emulator daemon logs the same lines it logs for a target that works,
//! the workload runs, sees a CPU agent and no GPU, and exits 0. Nothing
//! anywhere says the target was the problem, and the obvious reading of
//! that evidence — that the emulator is broken for this GPU — is wrong
//! and expensive: it cost about a day of investigation, and was very
//! nearly filed as an emulator bug (issue #11361).
//!
//! So mirage asks the question before the session starts. Which ROCm
//! runtime will this workload load, and does it know the ISA the profile
//! emulates?
//!
//! # Why the answer is read out of the library's own image
//!
//! There is no API for it. `hsa_isa_from_name` would answer exactly this
//! question and cannot be called: it needs `hsa_init()` first, which
//! needs the KFD device that does not exist yet, inside the session that
//! has not started. What there *is* is the table itself — ROCr carries
//! every ISA name it supports as a string in its own binary, which is
//! how the ISA list on a host is read in practice — so that is what this
//! reads.
//!
//! That is evidence rather than an interface, and this module is built
//! to fail towards silence because of it. Three outcomes, not two: the
//! target is named, the target is absent from a table that plainly *is*
//! a table, or no verdict at all — no runtime found, unreadable, or a
//! set of names too small to be ROCr's list. Only the middle one says
//! anything, and even then it warns rather than refuses, because a
//! wrong "unsupported" must never be able to block a run that would have
//! worked.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use crate::discovery::{self, LibSearch};

/// The ROCm runtime, under the SONAME the dynamic loader actually
/// resolves. This is the file a workload ends up with; the unversioned
/// name below is a development symlink beside it that a runtime-only
/// install need not have.
pub const ROCR_SONAME: &str = "libhsa-runtime64.so.1";

/// The unversioned development name for [`ROCR_SONAME`], tried second.
pub const ROCR_LIB: &str = "libhsa-runtime64.so";

/// How many plausible ISA names must be found in a library's image
/// before the absence of one more is worth reporting.
///
/// The floor is what separates "this runtime does not support the
/// target" from "this scan did not find ROCr's table". A stripped,
/// packed, or simply unexpected build yields a handful of fragments at
/// most; a real ROCm runtime carries every target it was built for, and
/// that has been dozens for as long as there have been dozens — the
/// ROCm 7.0.2.2 the bug was found on names forty-five. Ten is far below
/// any real table and far above any accident.
const MIN_PLAUSIBLE_TABLE: usize = 10;

/// The shortest run of characters after `gfx` that can be a whole
/// target name.
///
/// `gfx942`, `gfx90a` and `gfx1250` are three and four; `gfx9`, `gfx11`
/// and `gfx1f` are the architecture-family prefixes and wildcard
/// patterns ROCr also carries, and are not targets anybody emulates.
/// Dropping them keeps the count in [`MIN_PLAUSIBLE_TABLE`] honest.
const MIN_TARGET_DIGITS: usize = 3;

/// What the ROCm runtime on this host has to say about one GPU target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetSupport {
    /// The located runtime names this target, so it will enumerate the
    /// emulated device.
    Supported,
    /// The located runtime carries a target table and this target is not
    /// in it. The workload will see no GPU.
    Unsupported(Box<UnsupportedTarget>),
    /// No verdict: no ROCm runtime was found where mirage looked, it
    /// could not be read, or what was read does not look like a target
    /// table. Nothing is said to the user on this path.
    Unknown,
}

/// A target the ROCm runtime a session will load does not know about,
/// with everything needed to say so usefully.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedTarget {
    /// The ISA the profile emulates, e.g. `gfx1250`.
    pub target: String,
    /// The ROCm runtime mirage found, which is the one the workload is
    /// expected to load.
    pub runtime: PathBuf,
    /// The ROCm version recorded beside that runtime, when there is one.
    pub version: Option<String>,
    /// How many targets the runtime does name. Quoting it is what makes
    /// the claim checkable: a user who reads "names forty-five targets"
    /// knows a real table was found and one entry was missing from it,
    /// not that a scan came up empty and blamed their profile.
    pub known: usize,
}

impl UnsupportedTarget {
    /// The sentence mirage prints, in the shape the rest of its notices
    /// use: what is wrong, what it will look like, and the way out.
    ///
    /// The symptom is spelled out rather than left implied. "ROCr does
    /// not support gfx1250" is only actionable to somebody who already
    /// knows what ROCr does with an agent it cannot name, and the whole
    /// reason this exists is that the visible behaviour — a session that
    /// starts, a command that runs, an exit status of 0 — looks like
    /// success.
    #[must_use]
    pub fn explain(&self) -> String {
        let version = self
            .version
            .as_deref()
            .map_or_else(String::new, |v| format!(" (ROCm {v})"));
        format!(
            "the ROCm runtime this session will load does not support {target}, which is \
             the GPU this profile emulates. {runtime}{version} names {known} targets and \
             {target} is not one of them, so it will skip the emulated device: the \
             workload will see no GPU at all and is likely to still exit 0. Run it under \
             a ROCm that knows {target} — `--image <a newer ROCm image>` is the usual \
             way — or use a profile whose target this ROCm supports.",
            target = self.target,
            runtime = self.runtime.display(),
            known = self.known,
        )
    }
}

/// Whether the ROCm runtime this host would load knows `target`.
///
/// `target` is a conventional gfx name as
/// [`crate::hardware::gfx_name`] renders one, e.g. `gfx1250`.
#[must_use]
pub fn check_target(target: &str) -> TargetSupport {
    let Some(runtime) = locate() else {
        return TargetSupport::Unknown;
    };
    let Ok(image) = std::fs::read(&runtime) else {
        return TargetSupport::Unknown;
    };
    verdict(target, &runtime, rocm_version_beside(&runtime), &image)
}

/// The ROCm runtime a workload started here would load, as far as mirage
/// can tell.
///
/// This is the shared discovery policy — `$LD_LIBRARY_PATH`,
/// `$ROCM_HOME`/`$ROCM_PATH`, the ROCm SDK root, the standard system
/// directories — which is the loader's search as closely as anything
/// that is not the loader can be. It is not the loader, and the gap is
/// why every caller treats a miss as [`TargetSupport::Unknown`] and why
/// the message names the file that was actually read.
#[must_use]
pub fn locate() -> Option<PathBuf> {
    // The SONAME first: that is the name a `DT_NEEDED` resolves and so
    // the file a workload really gets. The unversioned symlink is a
    // development convenience that a runtime-only install may not ship,
    // and looking for it alone would report no ROCm on a host that has
    // one.
    discovery::find_emulator_lib(&search(ROCR_SONAME))
        .or_else(|| discovery::find_emulator_lib(&search(ROCR_LIB)))
}

/// The environment variables that decide which ROCm runtime [`locate`]
/// finds, and therefore which one any verdict here is about.
///
/// Derived from the search rather than written down, so a change to the
/// discovery policy cannot leave a stale list behind.
///
/// For a caller that is about to hand the workload a *different* value
/// for one of these — `mirage run --env LD_LIBRARY_PATH=…` outranks what
/// mirage itself inherited — because then the runtime this process can
/// reach is not the runtime the workload will load, and the honest
/// answer is [`TargetSupport::Unknown`].
#[must_use]
pub fn search_vars() -> Vec<String> {
    search(ROCR_SONAME)
        .env_hints()
        .into_iter()
        .map(|hint| hint.name)
        .collect()
}

/// The discovery policy for the ROCm runtime.
///
/// No override variables of its own: this is not a mirage backend a user
/// installs and points at, it is whatever ROCm the workload's loader will
/// find, and inventing a `MIRAGE_ROCR_LIB` would let the two disagree.
/// `system_fallbacks` is what makes the search the loader's.
fn search(lib_name: &'static str) -> LibSearch<'static> {
    LibSearch {
        file_env: &[],
        dir_env: &[],
        home_env: &[],
        lib_name,
        binary_relative_dirs: &[],
        system_fallbacks: true,
    }
}

/// Decide what `image` says about `target`, given where it came from.
///
/// Split from [`check_target`] so the rule can be tested against a
/// fabricated library: the question is about the ROCm on the host, which
/// a test cannot choose, and a test that could only assert about the
/// machine it happened to run on would assert nothing on most of them.
fn verdict(target: &str, runtime: &Path, version: Option<String>, image: &[u8]) -> TargetSupport {
    let known = target_names(image);
    if known.len() < MIN_PLAUSIBLE_TABLE {
        // Whatever this file is, its target table is not what was read,
        // and a "your ROCm does not support this GPU" drawn from a failed
        // scan would be a confident lie.
        return TargetSupport::Unknown;
    }
    if mentions(image, target) {
        return TargetSupport::Supported;
    }
    TargetSupport::Unsupported(Box::new(UnsupportedTarget {
        target: target.to_string(),
        runtime: runtime.to_path_buf(),
        version,
        known: known.len(),
    }))
}

/// Every plausible gfx target name in `image`.
///
/// Used for the count in the report and for the sanity floor, and
/// deliberately *not* for the membership test — see [`mentions`], which
/// is the forgiving half of the pair. This half is the strict one: a
/// name is only counted when it starts a word and is `gfx` followed by
/// nothing but lowercase hex digits, so the fragments a binary scan
/// inevitably turns up cannot inflate the table it is judged against.
fn target_names(image: &[u8]) -> BTreeSet<String> {
    const TAG: &[u8] = b"gfx";
    let mut out = BTreeSet::new();
    let mut at = 0;
    while at + TAG.len() <= image.len() {
        if &image[at..at + TAG.len()] != TAG {
            at += 1;
            continue;
        }
        let mut end = at + TAG.len();
        while end < image.len() && image[end].is_ascii_alphanumeric() {
            end += 1;
        }
        // `gfx` inside a longer word is somebody else's identifier, not
        // a target name.
        let starts_a_word = at == 0 || !is_word_byte(image[at - 1]);
        if starts_a_word && let Some(name) = plausible_target(&image[at + TAG.len()..end]) {
            out.insert(name);
        }
        at = end.max(at + 1);
    }
    out
}

/// The target name `digits` completes, if it can be one.
fn plausible_target(digits: &[u8]) -> Option<String> {
    if digits.len() < MIN_TARGET_DIGITS || !digits[0].is_ascii_digit() {
        return None;
    }
    if !digits
        .iter()
        .all(|b| b.is_ascii_digit() || b.is_ascii_lowercase() && *b <= b'f')
    {
        return None;
    }
    let digits = std::str::from_utf8(digits).ok()?;
    Some(format!("gfx{digits}"))
}

/// Whether `image` names `target` anywhere at all.
///
/// The membership test, and deliberately looser than [`target_names`].
/// Being wrong here in the generous direction costs nothing — mirage
/// stays quiet about a session that was going to work anyway — while
/// being wrong in the strict direction warns a user off a run that was
/// fine. So the only thing insisted on is that the match is not the
/// prefix of a longer target: `gfx1200` must not answer for `gfx120`.
fn mentions(image: &[u8], target: &str) -> bool {
    let needle = target.as_bytes();
    if needle.is_empty() {
        return false;
    }
    image.windows(needle.len()).enumerate().any(|(at, window)| {
        window == needle
            && image
                .get(at + needle.len())
                .is_none_or(|next| !next.is_ascii_alphanumeric())
    })
}

/// Whether `byte` can be part of an identifier, and so cannot precede
/// the start of one.
fn is_word_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

/// The ROCm version recorded beside the runtime at `lib`, if any.
///
/// A ROCm install writes its version to `<root>/.info/version`, and
/// `lib` is `<root>/lib/libhsa-runtime64.so.1`. Best effort: the version
/// is the most recognisable thing to say about a ROCm install and the
/// least essential, so a layout that does not have it costs the report a
/// parenthesis and nothing else.
fn rocm_version_beside(lib: &Path) -> Option<String> {
    let root = lib.parent()?.parent()?;
    let version = std::fs::read_to_string(root.join(".info").join("version")).ok()?;
    let version = version.trim();
    (!version.is_empty()).then(|| version.to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// A stand-in for a ROCm runtime's image: the targets it was built
    /// for, laid out the way an ELF string table lays them out.
    ///
    /// Interleaved with the architecture-family prefixes and the
    /// wildcard patterns a real `libhsa-runtime64.so` carries beside the
    /// real names, because those are what a naive scan miscounts as
    /// targets and the floor in [`MIN_PLAUSIBLE_TABLE`] exists to be
    /// judged against real names only.
    fn rocr_image(targets: &[&str]) -> Vec<u8> {
        let mut out: Vec<u8> = b"\x7fELF\0\0\0\0some other rodata\0".to_vec();
        for noise in ["gfx9", "gfx10", "gfx11", "gfx12", "gfx9f", "gfx1f"] {
            out.extend_from_slice(noise.as_bytes());
            out.push(0);
        }
        for target in targets {
            out.extend_from_slice(format!("amdgcn-amd-amdhsa--{target}").as_bytes());
            out.push(0);
            out.extend_from_slice(target.as_bytes());
            out.push(0);
        }
        out
    }

    /// A slice of what a ROCm 7.0 runtime names — every target the seven
    /// builtin profiles ask for, plus enough of the rest to clear
    /// [`MIN_PLAUSIBLE_TABLE`]. The real list is longer (forty-five on
    /// the host the bug was found on); the point of the fixture is which
    /// names are in it and which one is not.
    const ROCM_7: &[&str] = &[
        "gfx900", "gfx902", "gfx904", "gfx906", "gfx908", "gfx90a", "gfx90c", "gfx942", "gfx950",
        "gfx1010", "gfx1030", "gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201",
    ];

    fn check(target: &str, image: &[u8]) -> TargetSupport {
        verdict(
            target,
            Path::new("/opt/rocm/lib/libhsa-runtime64.so.1"),
            Some("7.0.2.2".to_string()),
            image,
        )
    }

    /// The regression, in the shape it actually happened: seven profiles
    /// on a ROCm 7.0 host, and the only one that enumerates no GPU is the
    /// only one whose target the runtime was not built for.
    #[test]
    fn the_one_target_the_runtime_does_not_name_is_the_one_reported() {
        let image = rocr_image(ROCM_7);
        for supported in [
            "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201",
        ] {
            assert_eq!(
                check(supported, &image),
                TargetSupport::Supported,
                "{supported} is in the table and must not be warned about"
            );
        }

        let TargetSupport::Unsupported(problem) = check("gfx1250", &image) else {
            panic!("gfx1250 is not in the table and must be reported");
        };
        assert_eq!(problem.target, "gfx1250");
        assert_eq!(problem.known, ROCM_7.len());
    }

    /// The message is the whole of the fix, so it is asserted on rather
    /// than assumed. It has to name the target, the file the verdict was
    /// read out of, the ROCm version, and what the failure will look
    /// like — because what it looks like is success.
    #[test]
    fn the_report_names_the_target_the_runtime_and_the_symptom() {
        let TargetSupport::Unsupported(problem) = check("gfx1250", &rocr_image(ROCM_7)) else {
            panic!("gfx1250 must be reported");
        };
        let message = problem.explain();

        assert!(message.contains("gfx1250"), "{message}");
        assert!(message.contains("libhsa-runtime64.so.1"), "{message}");
        assert!(message.contains("ROCm 7.0.2.2"), "{message}");
        // The symptom, spelled out: a session that starts, a command that
        // runs and an exit status of 0 is what a working run looks like
        // too, and that is why this took a day to diagnose.
        assert!(message.contains("no GPU"), "{message}");
        assert!(message.contains("exit 0"), "{message}");
        // And a way out.
        assert!(message.contains("--image"), "{message}");
    }

    /// A runtime with no version file beside it still gets a report,
    /// just without the parenthesis.
    #[test]
    fn a_missing_rocm_version_costs_the_report_nothing_else() {
        let TargetSupport::Unsupported(problem) = verdict(
            "gfx1250",
            Path::new("/somewhere/libhsa-runtime64.so.1"),
            None,
            &rocr_image(ROCM_7),
        ) else {
            panic!("gfx1250 must be reported");
        };
        let message = problem.explain();
        assert!(
            message.contains("/somewhere/libhsa-runtime64.so.1"),
            "{message}"
        );
        assert!(!message.contains("ROCm )"), "{message}");
        assert!(!message.contains("()"), "{message}");
    }

    /// Anything that is not recognisably a target table says nothing.
    ///
    /// This is the half that keeps a heuristic honest. The verdict is
    /// read out of a binary rather than asked of an API, so the scan can
    /// simply fail — a stripped build, a layout nobody anticipated, a
    /// file that is not ROCr at all — and a "your ROCm does not support
    /// this GPU" drawn from a failed scan would send a user to fix
    /// something that was never wrong.
    #[test]
    fn a_file_with_no_target_table_yields_no_verdict() {
        for image in [
            &b""[..],
            &b"\x7fELF not really a runtime"[..],
            // A couple of real names is not a table: ROCr carries dozens.
            &rocr_image(&["gfx942", "gfx950"])[..],
            // Neither are the family prefixes and wildcards on their own,
            // which is exactly what a scan that counted everything
            // `gfx`-shaped would mistake for one.
            &rocr_image(&[])[..],
        ] {
            assert_eq!(
                check("gfx1250", image),
                TargetSupport::Unknown,
                "a scan that found no table must not produce a verdict"
            );
        }
    }

    /// The prefixes and wildcards ROCr carries beside the real names are
    /// not targets, and counting them would let a stripped library clear
    /// the floor that exists to catch it.
    #[test]
    fn family_prefixes_and_wildcards_are_not_counted_as_targets() {
        let names = target_names(&rocr_image(&["gfx942", "gfx1250"]));
        assert_eq!(
            names,
            ["gfx1250".to_string(), "gfx942".to_string()]
                .into_iter()
                .collect::<BTreeSet<_>>()
        );
    }

    /// A name is only a name when it starts a word: `libgfx1250stuff` is
    /// somebody's symbol, not a supported target.
    #[test]
    fn a_gfx_inside_a_longer_identifier_is_not_a_target() {
        assert!(target_names(b"\0libgfx1250_handler\0").is_empty());
        assert_eq!(target_names(b"\0-gfx1250\0").len(), 1);
    }

    /// The membership test must not let a longer target answer for a
    /// shorter one. `gfx1200` in the table says nothing about `gfx120`.
    #[test]
    fn a_longer_target_does_not_answer_for_a_shorter_one() {
        let image = rocr_image(ROCM_7);
        assert!(mentions(&image, "gfx1200"));
        assert!(!mentions(&image, "gfx120"));
        assert!(!mentions(&image, ""));
    }

    /// The two names are tried in the order the loader would care about:
    /// the SONAME is the file a `DT_NEEDED` actually resolves, and the
    /// unversioned one is a development symlink a runtime-only install
    /// need not ship.
    #[test]
    fn the_soname_is_what_is_looked_for_first() {
        assert_eq!(search(ROCR_SONAME).lib_name, ROCR_SONAME);
        assert!(
            search(ROCR_SONAME).system_fallbacks,
            "the search must be the loader's, not mirage's own"
        );
        assert!(
            search(ROCR_SONAME).file_env.is_empty(),
            "an override here could only ever disagree with the loader"
        );
        // Both names are searched, and the versioned one is the primary.
        assert_eq!(ROCR_SONAME, format!("{ROCR_LIB}.1"));
    }

    /// The variables a caller has to watch for are the ones the search
    /// honours, and they are the loader's rather than mirage's own — a
    /// run that sets one of these is loading a different ROCm than this
    /// process would, and a verdict about the wrong library is worse
    /// than none.
    #[test]
    fn the_variables_that_redirect_the_search_are_the_loaders() {
        let vars = search_vars();
        for expected in ["LD_LIBRARY_PATH", "ROCM_HOME", "ROCM_PATH"] {
            assert!(vars.iter().any(|v| v == expected), "{vars:?}");
        }
        // And nothing of mirage's own invention, which is the same rule
        // `search` is written to: an override here could only ever
        // disagree with the loader.
        assert!(!vars.iter().any(|v| v.starts_with("MIRAGE_")), "{vars:?}");
    }

    /// Whatever this host has, asking must be safe and must agree with
    /// itself. The verdict is not asserted on: it is a fact about the
    /// machine the test runs on.
    #[test]
    fn asking_about_this_host_is_harmless() {
        let located = locate();
        let support = check_target("gfx1250");
        if located.is_none() {
            assert_eq!(support, TargetSupport::Unknown);
        }
        // A target no ROCm will ever name, on a host that has one, must
        // be reported rather than passed over.
        if let Some(path) = located
            && matches!(check_target("gfx942"), TargetSupport::Supported)
        {
            let TargetSupport::Unsupported(problem) = check_target("gfx0ff") else {
                panic!("a target no runtime names must be reported");
            };
            assert_eq!(problem.runtime, path);
        }
    }
}
