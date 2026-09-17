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
//! So mirage asks the question before each workload starts. Which ROCm
//! runtime does this command's dynamic loader resolve, and can its image
//! rule out the ISA the profile emulates?
//!
//! # Why the answer is read out of the library's own image
//!
//! There is no side-effect-free API for it. `hsa_isa_from_name` would
//! answer exactly this question but needs `hsa_init()` first, which would
//! initialise ROCr in mirage's own process rather than inspect the
//! library and environment selected for the workload. What the binary
//! does guarantee is one-way: every ISA in the registry leaves its
//! target name in the image. Other ROCr subsystems leave target names
//! there too, so presence is inconclusive, but absence from a
//! recognisably complete ROCr image rules support out.
//!
//! That is evidence rather than an interface, and this module is built
//! to fail towards silence because of it. A target name appearing in
//! the image proves nothing: ROCr also embeds names used only for legacy
//! code-object conversion. Only absence from a recognisably complete
//! image is a verdict. A missing loader answer, an unreadable runtime, a
//! target mention, or too few target references all yield no verdict.
//! Even a verdict warns rather than refuses, because this diagnostic
//! must never block a run that would have worked.

use std::collections::BTreeSet;
use std::ffi::{OsStr, OsString};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use goblin::Object;
use goblin::elf::dynamic::{DT_AUDIT, DT_DEPAUDIT};
use nix::unistd::{AccessFlags, eaccess};
use rustix::fs::getxattr;
use rustix::io::Errno;

/// The ROCm runtime SONAME the dynamic loader actually resolves. Unlike
/// the unversioned development symlink, this name exists in a
/// runtime-only install and is what a workload's `DT_NEEDED` requests.
pub const ROCR_SONAME: &str = "libhsa-runtime64.so.1";

/// How many plausible ISA names must be found in a library's image
/// before the absence of one more is worth reporting.
///
/// The floor is what separates "this runtime does not support the
/// target" from "this scan did not find ROCr's target-bearing image". A
/// stripped, packed, or simply unexpected build yields a handful of
/// fragments at most; a real ROCm runtime references dozens of targets
/// across its ISA registry and code-object conversion support — ROCm
/// 7.0.2.2 references forty-five. Ten is far below a real image and far
/// above an accident.
const MIN_PLAUSIBLE_TARGET_REFERENCES: usize = 10;

/// The shortest run of characters after `gfx` that can be a whole
/// target name.
///
/// `gfx942`, `gfx90a` and `gfx1250` are three and four; `gfx9`, `gfx11`
/// and `gfx1f` are the architecture-family prefixes and wildcard
/// patterns ROCr also carries, and are not targets anybody emulates.
/// Dropping them keeps the count in
/// [`MIN_PLAUSIBLE_TARGET_REFERENCES`] honest.
const MIN_TARGET_DIGITS: usize = 3;

/// What the ROCm runtime on this host has to say about one GPU target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetSupport {
    /// The resolved runtime is recognisably ROCr and contains no
    /// reference to this target. The workload will see no GPU.
    Unsupported(Box<UnsupportedTarget>),
    /// No verdict: the command's loader did not identify one runtime,
    /// loader-affecting state makes the answer ambiguous, the runtime
    /// could not be read, its image is not recognisably ROCr, or it
    /// mentions the target without proving that the ISA registry
    /// contains it. Nothing is said to the user on this path.
    Unknown,
}

/// A target the ROCm runtime a workload will load does not know about,
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
    /// How many distinct target names the runtime image references.
    /// Quoting it makes the heuristic checkable without claiming that
    /// every reference is an ISA-registry entry.
    pub references: usize,
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
    ///
    /// It says "this session" rather than "this profile" because those
    /// are not always the same answer: a drop-in `--config` names a
    /// device of its own, and the ISA reported here is read from the
    /// configuration the session was actually brought up on. Naming the
    /// profile would send a user to edit the one input that is not
    /// deciding this.
    #[must_use]
    pub fn explain(&self) -> String {
        let version = self
            .version
            .as_deref()
            .map_or_else(String::new, |v| format!(" (ROCm {v})"));
        format!(
            "the ROCm runtime this workload will load does not support {target}, which is \
             the GPU this session emulates. {runtime}{version} contains references to \
             {references} GPU targets but none to {target}, so it will skip the emulated device: the \
             workload will see no GPU at all and is likely to still exit 0. Run it under \
             a ROCm that knows {target} — `--image <a newer ROCm image>` is the usual \
             way — or emulate a GPU this ROCm supports.",
            target = self.target,
            runtime = self.runtime.display(),
            references = self.references,
        )
    }
}

/// Whether the ROCm runtime resolved for one host process rules out
/// `target`.
///
/// `env` and `inherit_env` are the exact environment delta and
/// inheritance policy the process spawner will use. The command is
/// resolved through that effective `PATH` and inspected statically,
/// without loading or executing any object. A verdict is produced only
/// for a trusted system executable that directly needs ROCr and resolves
/// it through an unambiguous modern `DT_RUNPATH`.
///
/// Legacy `DT_RPATH`, transitive or `dlopen` dependencies, cache-only
/// resolution, hardware-capability alternatives, mutable executables,
/// privileged executables, a system-wide preload, and loader overrides
/// all produce [`TargetSupport::Unknown`]. These deliberate false
/// negatives keep warnings trustworthy. Extending coverage requires a
/// bounded, loader-equivalent static resolver and can be done later
/// without weakening the rule that preflight never executes workload
/// code or emits a guess.
#[must_use]
pub fn check_target_for_process(
    target: &str,
    command: &str,
    workdir: Option<&Path>,
    env: &[(OsString, OsString)],
    inherit_env: bool,
) -> TargetSupport {
    let effective = merged_environment(std::env::vars_os(), env, inherit_env);
    if global_preload_is_ambiguous(Path::new(GLOBAL_PRELOAD))
        || loader_state_is_ambiguous(&effective)
    {
        return TargetSupport::Unknown;
    }
    let Some(runtime) = locate_for_process(command, workdir, &effective) else {
        return TargetSupport::Unknown;
    };
    check_target_at(target, &runtime)
}

/// Whether one already-resolved ROCm runtime rules out `target`.
///
/// Private: the size cap that makes reading the image safe is applied
/// while the loader trace selects it, so this is the second half of
/// [`check_target_for_process`] rather than an entry point of its own.
fn check_target_at(target: &str, runtime: &Path) -> TargetSupport {
    let Ok(image) = std::fs::read(runtime) else {
        return TargetSupport::Unknown;
    };
    verdict(target, runtime, rocm_version_beside(runtime), &image)
}

fn merged_environment(
    inherited: impl IntoIterator<Item = (OsString, OsString)>,
    explicit: &[(OsString, OsString)],
    inherit_env: bool,
) -> std::collections::BTreeMap<OsString, OsString> {
    let mut effective = if inherit_env {
        inherited.into_iter().collect()
    } else {
        std::collections::BTreeMap::new()
    };
    effective.extend(explicit.iter().cloned());
    effective
}

/// The file glibc preloads into every dynamically linked process on the
/// host, whatever its environment.
const GLOBAL_PRELOAD: &str = "/etc/ld.so.preload";

/// Whether a system-wide preload could be interposing ROCr.
///
/// `LD_PRELOAD` is the environment's half of this; `/etc/ld.so.preload`
/// is the host's, and no workload environment mentions it. A library
/// listed there is loaded before the workload's own dependencies and can
/// interpose the HSA entry points the target table is evidence about, so
/// the file's own image stops being the answer.
///
/// Only an outright absent file is harmless. Present and empty says
/// nothing is preloaded; unreadable says this probe could not find out,
/// which is not the same as finding out that there is nothing.
fn global_preload_is_ambiguous(path: &Path) -> bool {
    match std::fs::read(path) {
        Ok(bytes) => bytes.iter().any(|byte| !byte.is_ascii_whitespace()),
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// State that can replace ROCr or change the ISA ROCr evaluates without
/// changing the loader's ordinary SONAME answer.
fn loader_state_is_ambiguous(env: &std::collections::BTreeMap<OsString, OsString>) -> bool {
    env.iter().any(|(key, value)| {
        let Some(key) = key.to_str() else {
            return false;
        };
        // A search-list variable is ambiguous as soon as it is set at
        // all, empty or not. An empty *element* of a search list means
        // the current directory to glibc, so the meaning of an empty
        // whole value is a question about one loader's parsing rather
        // than about the workload — and this module answers questions it
        // is sure of or stays quiet.
        if loader_resolution_env_var(key) {
            return true;
        }
        // The rest name something to load or a version to report, so an
        // empty value asks for nothing.
        !value.is_empty() && ambiguous_probe_env_var(key)
    })
}

/// Whether changing `key` can change this process probe's answer.
///
/// Used by the multi-rank caller to deduplicate only specs that resolve
/// the same command and runtime. Keep the exact loader inputs, the
/// variables that force an unknown verdict, and `PATH`, which selects
/// the command whose dependencies are traced.
#[must_use]
pub fn affects_process_probe(key: &str) -> bool {
    key == "PATH" || loader_resolution_env_var(key) || ambiguous_probe_env_var(key)
}

fn ambiguous_probe_env_var(key: &str) -> bool {
    matches!(key, "LD_PRELOAD" | "LD_AUDIT" | "HSA_OVERRIDE_GFX_VERSION")
        || key.starts_with("HSA_OVERRIDE_GFX_VERSION_")
}

fn loader_resolution_env_var(key: &str) -> bool {
    matches!(
        key,
        "LD_LIBRARY_PATH"
            | "LD_ORIGIN_PATH"
            | "LD_HWCAP_MASK"
            | "LD_ASSUME_KERNEL"
            | "GLIBC_TUNABLES"
    )
}

fn locate_for_process(
    command: &str,
    workdir: Option<&Path>,
    effective: &std::collections::BTreeMap<OsString, OsString>,
) -> Option<PathBuf> {
    let workdir = absolute_workdir(workdir)?;
    let executable = resolve_command(command, Some(&workdir), effective)?;
    let executable = trusted_system_file(&executable)?;
    unprivileged_executable(&executable)?;
    let metadata = executable.metadata().ok()?;
    (metadata.len() <= 256 * 1024 * 1024).then_some(())?;
    let image = std::fs::read(&executable).ok()?;
    let runtime = direct_runpath_runtime(&executable, &image, true)?;
    trusted_system_file(&runtime)
}

fn direct_runpath_runtime(
    executable: &Path,
    image: &[u8],
    require_trusted_paths: bool,
) -> Option<PathBuf> {
    let Object::Elf(elf) = Object::parse(image).ok()? else {
        return None;
    };
    // An executable's own audit tags install callbacks without any
    // LD_AUDIT environment variable; la_objsearch may replace the
    // runtime before RUNPATH resolution.
    if !elf.rpaths.is_empty()
        || !elf.libraries.contains(&ROCR_SONAME)
        || elf.dynamic.as_ref().is_some_and(|dynamic| {
            dynamic
                .dyns
                .iter()
                .any(|entry| matches!(entry.d_tag, DT_AUDIT | DT_DEPAUDIT))
        })
    {
        return None;
    }

    let origin = executable.parent()?.display().to_string();
    for entry in &elf.runpaths {
        for directory in entry.split(':') {
            if directory.is_empty() {
                return None;
            }
            let directory = expand_origin(directory, &origin);
            if directory.contains('$') {
                return None;
            }
            let mut directory = PathBuf::from(directory);
            if !directory.is_absolute() {
                return None;
            }
            // A `RUNPATH` entry that is not there is not ambiguous: the
            // loader finds nothing in it and moves on to the next entry,
            // and so does this. Anything else about it — unreadable,
            // untrusted — is a directory the loader *will* search and
            // this cannot vouch for, which is a verdict declined rather
            // than an entry skipped.
            match std::fs::symlink_metadata(&directory) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
                Err(_) => return None,
                Ok(_) => {}
            }
            if require_trusted_paths {
                directory = trusted_system_directory(&directory)?;
            }
            if alternative_runtime_exists(&directory, require_trusted_paths)? {
                return None;
            }
            let mut candidate = directory.join(ROCR_SONAME);
            if candidate.is_file() {
                if require_trusted_paths {
                    candidate = trusted_system_file(&candidate)?;
                }
                (candidate.metadata().ok()?.len() <= 512 * 1024 * 1024).then_some(())?;
                let candidate_image = std::fs::read(&candidate).ok()?;
                let Object::Elf(candidate_elf) = Object::parse(&candidate_image).ok()? else {
                    return None;
                };
                if candidate_elf.header.e_machine != elf.header.e_machine
                    || candidate_elf.is_64 != elf.is_64
                {
                    return None;
                }
                return std::fs::canonicalize(candidate).ok();
            }
        }
    }
    None
}

fn expand_origin(directory: &str, origin: &str) -> String {
    let directory = directory.replace("${ORIGIN}", origin);
    let mut expanded = String::with_capacity(directory.len());
    let mut rest = directory.as_str();
    while let Some(at) = rest.find("$ORIGIN") {
        expanded.push_str(&rest[..at]);
        let after = &rest[at + "$ORIGIN".len()..];
        if after
            .as_bytes()
            .first()
            .is_none_or(|byte| !byte.is_ascii_alphanumeric() && *byte != b'_')
        {
            expanded.push_str(origin);
        } else {
            expanded.push_str("$ORIGIN");
        }
        rest = after;
    }
    expanded.push_str(rest);
    expanded
}

/// Whether some other ROCr under `directory` could be selected ahead of
/// the one directly in it.
///
/// Two families of them, because glibc has had two. Modern glibc
/// augments a RUNPATH directory with `<dir>/glibc-hwcaps/<level>`.
/// Before 2.37 it also searched hardware-capability subdirectories
/// directly — `<dir>/tls`, `<dir>/x86_64`, `<dir>/haswell` and the
/// platform's own name — so a runtime in one of those is the one a
/// workload on such a host loads while this probe reads the base copy.
///
/// Neither family is searched for by name: the legacy names are
/// per-architecture and per-CPU and a list of them would be a second
/// thing to keep in step with glibc. Any immediate subdirectory holding
/// a ROCr is treated as an alternative instead, which needs no list and
/// costs one directory read.
///
/// What it is *not* is a recursive search. glibc does not look below
/// those directories, and walking every tree under `<dir>` both invents
/// candidates the loader cannot select and, on an ordinary ROCm install,
/// buries the real runtime under asset trees like `hipblaslt/library`.
/// The remaining gap is the legacy *combinations* — `<dir>/haswell/tls`
/// and friends, nested one level deeper — which no ROCm packaging ships
/// and which stay a documented false verdict rather than an unbounded
/// walk.
fn alternative_runtime_exists(directory: &Path, require_trusted_paths: bool) -> Option<bool> {
    let listing = match std::fs::read_dir(directory) {
        Ok(listing) => listing,
        // Gone between the check above and here: nothing in a directory
        // that is not there, and the caller moves on as the loader does.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(false),
        Err(_) => return None,
    };
    for entry in listing {
        let path = entry.ok()?.path();
        if !path.is_dir() {
            continue;
        }
        let found = if path.file_name() == Some(OsStr::new("glibc-hwcaps")) {
            hwcap_runtime_exists(&path, require_trusted_paths)?
        } else {
            // Only whether it is there. An alternative this probe cannot
            // rule out means no verdict either way, so nothing about it
            // is worth a trust check the selected runtime already gets.
            path.join(ROCR_SONAME).is_file()
        };
        if found {
            return Some(true);
        }
    }
    Some(false)
}

/// Whether a modern `<dir>/glibc-hwcaps/<level>` holds a ROCr.
fn hwcap_runtime_exists(hwcaps: &Path, require_trusted_paths: bool) -> Option<bool> {
    let mut hwcaps = hwcaps.to_path_buf();
    if require_trusted_paths {
        hwcaps = trusted_system_directory(&hwcaps)?;
    }
    let listing = match std::fs::read_dir(hwcaps) {
        Ok(listing) => listing,
        // The same race, and the same answer: no alternative here.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Some(false),
        Err(_) => return None,
    };
    for entry in listing {
        let entry = entry.ok()?;
        let mut path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if require_trusted_paths {
            path = trusted_system_directory(&path)?;
        }
        let runtime = path.join(ROCR_SONAME);
        if runtime.is_file() {
            if require_trusted_paths {
                trusted_system_file(&runtime)?;
            }
            return Some(true);
        }
    }
    Some(false)
}

fn trusted_system_file(path: &Path) -> Option<PathBuf> {
    trusted_system_path(path, true)
}

fn unprivileged_executable(path: &Path) -> Option<()> {
    let metadata = path.metadata().ok()?;
    (metadata.mode() & 0o6000 == 0).then_some(())?;
    let mut capabilities = [0_u8; 64];
    match getxattr(path, "security.capability", &mut capabilities[..]) {
        Ok(0) | Err(Errno::NODATA) => Some(()),
        Ok(_) | Err(_) => None,
    }
}

fn trusted_system_directory(path: &Path) -> Option<PathBuf> {
    trusted_system_path(path, false)
}

fn trusted_system_path(path: &Path, require_file: bool) -> Option<PathBuf> {
    trusted_path_chain(path, require_file)?;
    let path = std::fs::canonicalize(path).ok()?;
    trusted_path_chain(&path, require_file)?;
    Some(path)
}

fn trusted_path_chain(path: &Path, require_file: bool) -> Option<()> {
    for (index, component) in path.ancestors().enumerate() {
        let metadata = component.metadata().ok()?;
        if metadata.uid() != 0 || metadata.mode() & 0o022 != 0 {
            return None;
        }
        if index == 0 {
            let expected_kind = if require_file {
                metadata.is_file()
            } else {
                metadata.is_dir()
            };
            expected_kind.then_some(())?;
        }
    }
    Some(())
}

fn resolve_command(
    command: &str,
    workdir: Option<&Path>,
    env: &std::collections::BTreeMap<OsString, OsString>,
) -> Option<PathBuf> {
    let base = absolute_workdir(workdir)?;
    if command.contains('/') {
        let path = PathBuf::from(command);
        let path = if path.is_absolute() {
            path
        } else {
            base.join(path)
        };
        return is_executable_file(&path).then_some(path);
    }
    let path = env.get(OsStr::new("PATH"))?;
    std::env::split_paths(path).find_map(|dir| {
        let dir = if dir.is_absolute() {
            dir.to_path_buf()
        } else {
            base.join(dir)
        };
        let candidate = dir.join(command);
        is_executable_file(&candidate).then_some(candidate)
    })
}

fn absolute_workdir(workdir: Option<&Path>) -> Option<PathBuf> {
    match workdir {
        Some(path) if path.is_absolute() => Some(path.to_path_buf()),
        Some(path) => Some(std::env::current_dir().ok()?.join(path)),
        None => std::env::current_dir().ok(),
    }
}

fn is_executable_file(path: &Path) -> bool {
    path.metadata().is_ok_and(|metadata| metadata.is_file())
        && eaccess(path, AccessFlags::X_OK).is_ok()
}

/// Decide what `image` says about `target`, given where it came from.
///
/// Split from [`check_target_for_process`] so the rule can be tested
/// against a fabricated library: the question is about the ROCm on the
/// host, which a test cannot choose, and a test that could only assert
/// about the machine it happened to run on would assert nothing on most
/// of them.
fn verdict(target: &str, runtime: &Path, version: Option<String>, image: &[u8]) -> TargetSupport {
    let known = target_names(image);
    if known.len() < MIN_PLAUSIBLE_TARGET_REFERENCES {
        // Whatever this file is, a recognisable ROCr image is not what
        // was read, and a "your ROCm does not support this GPU" drawn
        // from a failed scan would be a confident lie.
        return TargetSupport::Unknown;
    }
    if mentions(image, target) {
        // ROCr contains names that its ISA registry does not: notably
        // gfx600/601/602 for legacy code-object conversion. A mention is
        // therefore not proof of support. Absence is still proof of
        // non-support once the image is recognisably complete.
        return TargetSupport::Unknown;
    }
    TargetSupport::Unsupported(Box::new(UnsupportedTarget {
        target: target.to_string(),
        runtime: runtime.to_path_buf(),
        version,
        references: known.len(),
    }))
}

/// Every plausible gfx target name in `image`.
///
/// Used for the count in the report and for the sanity floor. This is
/// the strict half of the scan: a name is only counted when the whole
/// word it sits in is `gfx` followed by nothing but lowercase hex
/// digits, so the fragments a binary scan inevitably turns up cannot
/// inflate the image it is judged against.
fn target_names(image: &[u8]) -> BTreeSet<String> {
    const TAG: &[u8] = b"gfx";
    let mut out = BTreeSet::new();
    let mut at = 0;
    while at + TAG.len() <= image.len() {
        if &image[at..at + TAG.len()] != TAG {
            at += 1;
            continue;
        }
        // The whole word, `_` included: stopping at `_` would read
        // `gfx942_helper` as a gfx942 reference, and enough identifiers
        // shaped like that would clear the floor below and condemn a
        // target the runtime supports.
        let mut end = at + TAG.len();
        while end < image.len() && is_word_byte(image[end]) {
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
/// Deliberately looser than [`target_names`]. Any mention makes the
/// result inconclusive, because a binary string does not say whether it
/// belongs to the ISA registry. The only thing insisted on is that the
/// match is not the prefix of a longer target: `gfx1200` must not answer
/// for `gfx120`. The word boundaries [`target_names`] insists on are
/// deliberately not insisted on here — `gfx942_helper` counts as a
/// mention of `gfx942` — because accepting more of them only produces
/// more silence.
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

    use std::os::unix::fs::PermissionsExt;

    use super::*;

    /// A stand-in for a ROCm runtime's image: the targets it was built
    /// for, laid out the way an ELF string table lays them out.
    ///
    /// Interleaved with the architecture-family prefixes and the
    /// wildcard patterns a real `libhsa-runtime64.so` carries beside the
    /// real names, because those are what a naive scan miscounts as
    /// targets and the floor in [`MIN_PLAUSIBLE_TARGET_REFERENCES`]
    /// exists to be judged against real names only.
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
    /// [`MIN_PLAUSIBLE_TARGET_REFERENCES`]. The real list is longer
    /// (forty-five on the host the bug was found on); the point of the
    /// fixture is which names are in it and which one is not.
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
    /// on a ROCm 7.0 host, and only the absent target can be condemned.
    /// A present string stays inconclusive because it may be conversion
    /// metadata rather than an ISA-registry entry.
    #[test]
    fn the_one_target_the_runtime_does_not_name_is_the_one_reported() {
        let image = rocr_image(ROCM_7);
        for mentioned in [
            "gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1151", "gfx1201",
        ] {
            assert_eq!(
                check(mentioned, &image),
                TargetSupport::Unknown,
                "{mentioned} is mentioned, but a raw string cannot prove registry support"
            );
        }

        let TargetSupport::Unsupported(problem) = check("gfx1250", &image) else {
            panic!("gfx1250 is absent from the image and must be reported");
        };
        assert_eq!(problem.target, "gfx1250");
        assert_eq!(problem.references, ROCM_7.len());
    }

    /// ROCr carries these full target names for legacy code-object
    /// conversion but does not register the corresponding ISAs. They
    /// are the concrete counterexample to treating a binary-string hit
    /// as support.
    #[test]
    fn legacy_conversion_names_do_not_count_as_registry_support() {
        let mut image = rocr_image(ROCM_7);
        for legacy in ["gfx600", "gfx601", "gfx602"] {
            image.extend_from_slice(format!("amdgcn-amd-amdhsa--{legacy}\0").as_bytes());
            assert_eq!(
                check(legacy, &image),
                TargetSupport::Unknown,
                "{legacy} is conversion metadata, not proof of ISA support"
            );
        }
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
        // The target is the session's, which for a drop-in `--config` is
        // not the profile's: naming the profile would point a user at
        // the input that is not deciding this.
        assert!(message.contains("this session emulates"), "{message}");
        assert!(!message.contains("profile"), "{message}");
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

    /// Anything that is not recognisably a ROCr target-bearing image
    /// says nothing.
    ///
    /// This is the half that keeps a heuristic honest. The verdict is
    /// read out of a binary rather than asked of an API, so the scan can
    /// simply fail — a stripped build, a layout nobody anticipated, a
    /// file that is not ROCr at all — and a "your ROCm does not support
    /// this GPU" drawn from a failed scan would send a user to fix
    /// something that was never wrong.
    #[test]
    fn an_unrecognisable_image_yields_no_verdict() {
        for image in [
            &b""[..],
            &b"\x7fELF not really a runtime"[..],
            // A couple of real names is not a recognisable image: ROCr
            // carries dozens.
            &rocr_image(&["gfx942", "gfx950"])[..],
            // Neither are the family prefixes and wildcards on their own,
            // which is exactly what a scan that counted everything
            // `gfx`-shaped would mistake for one.
            &rocr_image(&[])[..],
        ] {
            assert_eq!(
                check("gfx1250", image),
                TargetSupport::Unknown,
                "an unrecognisable image must not produce a verdict"
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

    /// A name is only a name when it is a whole word: `libgfx1250stuff`
    /// and `gfx942_helper` are somebody's symbols, not supported targets.
    #[test]
    fn a_gfx_inside_a_longer_identifier_is_not_a_target() {
        assert!(target_names(b"\0libgfx1250_handler\0").is_empty());
        assert_eq!(target_names(b"\0-gfx1250\0").len(), 1);
        assert!(target_names(b"\0gfx942_helper\0").is_empty());
        assert!(target_names(b"\0gfx942helper\0").is_empty());
    }

    /// Identifiers that merely begin with a target name cannot pass for
    /// a target table.
    ///
    /// The floor in [`MIN_PLAUSIBLE_TARGET_REFERENCES`] is the whole
    /// defence against condemning a target on the strength of a scan
    /// that never found ROCr's names. Ending a token at `_` would let
    /// ten identifiers like `gfx900_helper` clear it, and an arbitrary
    /// image would then report every target as unsupported.
    #[test]
    fn suffixed_identifiers_cannot_clear_the_confidence_floor() {
        let mut image: Vec<u8> = b"\x7fELF\0".to_vec();
        for n in 0..MIN_PLAUSIBLE_TARGET_REFERENCES + 5 {
            image.extend_from_slice(format!("gfx{}_helper\0", 900 + n).as_bytes());
        }

        assert!(target_names(&image).is_empty());
        assert_eq!(check("gfx1250", &image), TargetSupport::Unknown);
    }

    /// The membership test must not let a longer target answer for a
    /// shorter one. A `gfx1200` mention says nothing about `gfx120`.
    #[test]
    fn a_longer_target_does_not_answer_for_a_shorter_one() {
        let image = rocr_image(ROCM_7);
        assert!(mentions(&image, "gfx1200"));
        assert!(!mentions(&image, "gfx120"));
        assert!(!mentions(&image, ""));
    }

    /// PATH lookup has to follow executable candidates, as `execvp`
    /// does. Stopping at an earlier non-executable file could inspect a
    /// different program from the one the workload will run.
    #[test]
    fn command_resolution_skips_non_executable_path_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let blocked_dir = tmp.path().join("blocked");
        let runnable_dir = tmp.path().join("runnable");
        std::fs::create_dir_all(&blocked_dir).unwrap();
        std::fs::create_dir_all(&runnable_dir).unwrap();
        let blocked = blocked_dir.join("workload");
        let runnable = runnable_dir.join("workload");
        std::fs::write(&blocked, b"not executable").unwrap();
        std::fs::write(&runnable, b"executable").unwrap();
        // No execute bit is set, so both ordinary users and UID 0 reject
        // the candidate. Root treats any execute bit as sufficient for
        // X_OK, unlike the owner/group selection an ordinary user gets.
        std::fs::set_permissions(&blocked, std::fs::Permissions::from_mode(0o400)).unwrap();
        std::fs::set_permissions(&runnable, std::fs::Permissions::from_mode(0o755)).unwrap();

        let env = [(
            OsString::from("PATH"),
            std::env::join_paths([blocked_dir, runnable_dir]).unwrap(),
        )]
        .into_iter()
        .collect();
        assert_eq!(resolve_command("workload", None, &env), Some(runnable));
        assert_eq!(
            resolve_command(blocked.to_str().unwrap(), None, &env),
            None,
            "a command containing a slash must fail rather than fall through PATH"
        );
    }

    #[test]
    fn only_immutable_system_files_can_produce_a_warning() {
        assert!(trusted_system_file(Path::new("/bin/true")).is_some());

        let tmp = tempfile::tempdir().unwrap();
        let executable = tmp.path().join("workload");
        std::fs::copy("/bin/true", &executable).unwrap();
        assert_eq!(trusted_system_file(&executable), None);

        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o4755)).unwrap();
        assert_eq!(unprivileged_executable(&executable), None);
    }

    #[test]
    fn dependency_analysis_does_not_execute_transitive_audit_code() {
        use std::process::Command;

        if Command::new("cc").arg("--version").output().is_err() {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let marker = tmp.path().join("audit-ran");
        let audit_source = tmp.path().join("audit.c");
        let audit = tmp.path().join("libaudit.so");
        std::fs::write(
            &audit_source,
            format!(
                "#include <stdio.h>\n\
                 __attribute__((constructor)) static void run(void) {{ \
                 FILE *f = fopen(\"{}\", \"w\"); if (f) fclose(f); }}\n",
                marker.display()
            ),
        )
        .unwrap();
        let built_audit = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&audit_source)
            .arg("-o")
            .arg(&audit)
            .status()
            .unwrap();
        if !built_audit.success() {
            return;
        }

        let runtime_source = tmp.path().join("runtime.c");
        let runtime = tmp.path().join(ROCR_SONAME);
        std::fs::write(&runtime_source, "int hsa_stub(void) { return 0; }\n").unwrap();
        let built_runtime = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&runtime_source)
            .arg(format!("-Wl,-soname,{ROCR_SONAME}"))
            .arg(format!("-Wl,--audit,{}", audit.display()))
            .arg("-o")
            .arg(&runtime)
            .status()
            .unwrap();
        if !built_runtime.success() {
            return;
        }

        let workload_source = tmp.path().join("workload.c");
        let workload = tmp.path().join("workload");
        std::fs::write(
            &workload_source,
            "extern int hsa_stub(void); int main(void) { return hsa_stub(); }\n",
        )
        .unwrap();
        let built_workload = Command::new("cc")
            .arg(&workload_source)
            .arg("-L")
            .arg(tmp.path())
            .arg(format!("-l:{ROCR_SONAME}"))
            .arg("-Wl,--enable-new-dtags")
            .arg("-Wl,-rpath,$ORIGIN")
            .arg("-o")
            .arg(&workload)
            .status()
            .unwrap();
        if !built_workload.success() {
            return;
        }

        let workload_image = std::fs::read(&workload).unwrap();
        assert_eq!(
            direct_runpath_runtime(&workload, &workload_image, false),
            None,
            "the linker's propagated DT_DEPAUDIT must suppress the verdict"
        );
        assert!(
            !marker.exists(),
            "static dependency analysis must not load an audit module"
        );

        let built_runtime = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&runtime_source)
            .arg(format!("-Wl,-soname,{ROCR_SONAME}"))
            .arg("-o")
            .arg(&runtime)
            .status()
            .unwrap();
        assert!(built_runtime.success(), "linking the clean runtime");
        let built_workload = Command::new("cc")
            .arg(&workload_source)
            .arg("-L")
            .arg(tmp.path())
            .arg(format!("-l:{ROCR_SONAME}"))
            .arg("-Wl,--enable-new-dtags")
            .arg("-Wl,-rpath,$ORIGIN")
            .arg("-o")
            .arg(&workload)
            .status()
            .unwrap();
        assert!(built_workload.success(), "linking the clean workload");
        let workload_image = std::fs::read(&workload).unwrap();
        assert_eq!(
            direct_runpath_runtime(&workload, &workload_image, false),
            Some(runtime.canonicalize().unwrap())
        );

        let unrelated = tmp.path().join("hipblaslt").join("library");
        std::fs::create_dir_all(&unrelated).unwrap();
        std::os::unix::fs::symlink(&runtime, unrelated.join(ROCR_SONAME)).unwrap();
        assert_eq!(
            direct_runpath_runtime(&workload, &workload_image, false),
            Some(runtime.canonicalize().unwrap()),
            "a library below an unrelated asset directory is not a loader alternative"
        );

        let hwcap = tmp.path().join("glibc-hwcaps").join("x86-64-v3");
        std::fs::create_dir_all(&hwcap).unwrap();
        std::os::unix::fs::symlink(&runtime, hwcap.join(ROCR_SONAME)).unwrap();
        assert_eq!(
            direct_runpath_runtime(&workload, &workload_image, false),
            None,
            "a symlinked hardware-capability alternative makes the selected runtime ambiguous"
        );
        std::fs::remove_dir_all(tmp.path().join("glibc-hwcaps")).unwrap();

        // The other half of the same question, on the glibc most hosts
        // are still running: before 2.37 the hardware-capability
        // directories sat directly under the RUNPATH entry, so a runtime
        // in `tls` is the one such a loader picks while this probe reads
        // the base copy beside it.
        let legacy = tmp.path().join("tls");
        std::fs::create_dir_all(&legacy).unwrap();
        std::os::unix::fs::symlink(&runtime, legacy.join(ROCR_SONAME)).unwrap();
        assert_eq!(
            direct_runpath_runtime(&workload, &workload_image, false),
            None,
            "a pre-2.37 glibc searches <dir>/tls, so the selected runtime is ambiguous"
        );
        std::fs::remove_dir_all(&legacy).unwrap();
        assert_eq!(
            direct_runpath_runtime(&workload, &workload_image, false),
            Some(runtime.canonicalize().unwrap()),
            "and the verdict comes back once the alternative is gone"
        );

        // A `RUNPATH` entry that does not exist is not a reason to stop
        // looking. The loader finds nothing there and tries the next
        // entry; treating it as ambiguous would let one stale directory
        // silence the diagnostic for the whole list.
        let stale = tmp.path().join("gone");
        let multi_entry = tmp.path().join("workload-multi");
        let built = Command::new("cc")
            .arg(&workload_source)
            .arg("-L")
            .arg(tmp.path())
            .arg(format!("-l:{ROCR_SONAME}"))
            .arg("-Wl,--enable-new-dtags")
            .arg(format!("-Wl,-rpath,{}:$ORIGIN", stale.display()))
            .arg("-o")
            .arg(&multi_entry)
            .status()
            .unwrap();
        assert!(
            built.success(),
            "linking a workload with a stale first entry"
        );
        assert!(!stale.exists());
        let multi_image = std::fs::read(&multi_entry).unwrap();
        // Proven rather than assumed: a linker that dropped the missing
        // entry would leave this test passing on a single-entry RUNPATH,
        // which is not the case it exists for.
        let Object::Elf(elf) = Object::parse(&multi_image).unwrap() else {
            panic!("the linker produced an ELF");
        };
        assert!(
            elf.runpaths.iter().any(|entry| {
                let mut entries = entry.split(':');
                entries.next() == Some(stale.to_str().unwrap()) && entries.next().is_some()
            }),
            "the fixture needs the missing directory first and another entry after it: {:?}",
            elf.runpaths
        );
        assert_eq!(
            direct_runpath_runtime(&multi_entry, &multi_image, false),
            Some(runtime.canonicalize().unwrap()),
            "a missing RUNPATH entry must be skipped, not treated as ambiguous"
        );

        for tag in ["--audit", "--depaudit"] {
            let audited_workload = tmp
                .path()
                .join(format!("workload-{}", tag.trim_start_matches("--")));
            let built_workload = Command::new("cc")
                .arg(&workload_source)
                .arg("-L")
                .arg(tmp.path())
                .arg(format!("-l:{ROCR_SONAME}"))
                .arg("-Wl,--enable-new-dtags")
                .arg("-Wl,-rpath,$ORIGIN")
                .arg(format!("-Wl,{tag},{}", audit.display()))
                .arg("-o")
                .arg(&audited_workload)
                .status()
                .unwrap();
            assert!(built_workload.success(), "linking an executable with {tag}");
            assert_eq!(
                direct_runpath_runtime(
                    &audited_workload,
                    &std::fs::read(&audited_workload).unwrap(),
                    false
                ),
                None,
                "{tag} can redirect the runtime lookup and must suppress a verdict"
            );
        }
    }

    #[test]
    fn only_complete_origin_tokens_are_expanded() {
        assert_eq!(expand_origin("$ORIGIN/lib", "/opt/app"), "/opt/app/lib");
        assert_eq!(
            expand_origin("${ORIGIN}_extra", "/opt/app"),
            "/opt/app_extra"
        );
        assert_eq!(
            expand_origin("$ORIGIN_extra:/opt/rocm/lib", "/opt/app"),
            "$ORIGIN_extra:/opt/rocm/lib"
        );
    }

    #[test]
    fn a_relative_workdir_is_resolved_before_the_loader_changes_directory() {
        let current = std::env::current_dir().unwrap();
        assert_eq!(
            absolute_workdir(Some(Path::new("relative/work"))),
            Some(current.join("relative/work"))
        );
    }

    /// `--clear-env-vars` means ambient loader state cannot leak into
    /// the diagnostic any more than it can leak into the workload.
    #[test]
    fn a_cleared_environment_drops_ambient_loader_state() {
        let inherited = [
            (OsString::from("PATH"), OsString::from("/ambient/bin")),
            (
                OsString::from("LD_LIBRARY_PATH"),
                OsString::from("/ambient/rocm/lib"),
            ),
        ];
        let explicit = [(OsString::from("PATH"), OsString::from("/clean/bin"))];

        let inherited_env = merged_environment(inherited.clone(), &explicit, true);
        assert_eq!(
            inherited_env.get(OsStr::new("LD_LIBRARY_PATH")),
            Some(&OsString::from("/ambient/rocm/lib"))
        );

        let clean_env = merged_environment(inherited, &explicit, false);
        assert_eq!(clean_env.get(OsStr::new("LD_LIBRARY_PATH")), None);
        assert_eq!(
            clean_env.get(OsStr::new("PATH")),
            Some(&OsString::from("/clean/bin"))
        );
    }

    /// Preloads can replace the runtime's symbols, HSA overrides change
    /// which ISA is looked up, and loader-selection variables can
    /// select user-controlled transitive dependencies. None permits a
    /// safe verdict from a loader trace.
    #[test]
    fn preload_and_hsa_overrides_make_the_answer_ambiguous() {
        for key in [
            "LD_PRELOAD",
            "LD_AUDIT",
            "HSA_OVERRIDE_GFX_VERSION",
            "HSA_OVERRIDE_GFX_VERSION_0",
            "HSA_OVERRIDE_GFX_VERSION_17",
        ] {
            let env = [(OsString::from(key), OsString::from("set"))]
                .into_iter()
                .collect();
            assert!(
                loader_state_is_ambiguous(&env),
                "{key} must suppress a loader verdict"
            );
        }
        let unrelated = [(OsString::from("HSA_ENABLE_SDMA"), OsString::from("0"))]
            .into_iter()
            .collect();
        assert!(!loader_state_is_ambiguous(&unrelated));
    }

    /// `LD_PRELOAD` is the environment's half of the preload question
    /// and `/etc/ld.so.preload` is the host's, which no workload
    /// environment mentions and which glibc applies to every
    /// dynamically linked process on the machine.
    ///
    /// Absent is the ordinary case and the only harmless one. Present
    /// and blank preloads nothing; unreadable is not the same fact as
    /// "there is nothing there", and this module does not round the
    /// second into the first.
    #[test]
    fn a_system_wide_preload_makes_the_answer_ambiguous() {
        let tmp = tempfile::tempdir().unwrap();
        let preload = tmp.path().join("ld.so.preload");

        assert!(!global_preload_is_ambiguous(&preload));

        for blank in [&b""[..], b"\n", b"  \t\n \n"] {
            std::fs::write(&preload, blank).unwrap();
            assert!(!global_preload_is_ambiguous(&preload));
        }

        std::fs::write(&preload, b"/usr/lib/libinterpose.so\n").unwrap();
        assert!(global_preload_is_ambiguous(&preload));

        // A directory stands in for the file that is there and cannot be
        // read: the error is not `NotFound`, so nothing was ruled out.
        assert!(global_preload_is_ambiguous(tmp.path()));
    }

    /// Setting a loader search list to nothing is still setting it.
    ///
    /// glibc gives an empty *element* of a search list the meaning
    /// "the current directory", so what an empty whole value means is a
    /// question about one loader's parsing of one variable. This module
    /// does not answer those: the workload and the probe have to resolve
    /// the same file, and an explicitly emptied search list is not
    /// evidence that they will. The variables that name something to
    /// load or a version to report are different — empty asks for
    /// nothing — and must stay harmless, or a workload environment that
    /// merely mentions `LD_PRELOAD` would silence the diagnostic.
    #[test]
    fn an_emptied_search_list_is_ambiguous_but_an_empty_preload_is_not() {
        for key in [
            "LD_LIBRARY_PATH",
            "LD_ORIGIN_PATH",
            "LD_HWCAP_MASK",
            "LD_ASSUME_KERNEL",
            "GLIBC_TUNABLES",
        ] {
            let env = [(OsString::from(key), OsString::new())]
                .into_iter()
                .collect();
            assert!(
                loader_state_is_ambiguous(&env),
                "an emptied {key} is still a loader instruction"
            );
        }

        for key in [
            "LD_PRELOAD",
            "LD_AUDIT",
            "HSA_OVERRIDE_GFX_VERSION",
            "HSA_OVERRIDE_GFX_VERSION_0",
        ] {
            let env = [(OsString::from(key), OsString::new())]
                .into_iter()
                .collect();
            assert!(
                !loader_state_is_ambiguous(&env),
                "an empty {key} loads nothing and overrides nothing"
            );
        }
    }

    #[test]
    fn every_loader_input_is_deduplicated_and_redirects_are_ambiguous() {
        for key in [
            "PATH",
            "LD_LIBRARY_PATH",
            "LD_ORIGIN_PATH",
            "LD_HWCAP_MASK",
            "LD_ASSUME_KERNEL",
            "GLIBC_TUNABLES",
            "LD_PRELOAD",
            "LD_AUDIT",
            "HSA_OVERRIDE_GFX_VERSION_0",
        ] {
            assert!(
                affects_process_probe(key),
                "{key} can change the probe answer"
            );
        }
        assert!(
            !affects_process_probe("BASH_ENV"),
            "unrelated workload setup must not split deduplication"
        );

        for key in [
            "LD_LIBRARY_PATH",
            "LD_ORIGIN_PATH",
            "LD_HWCAP_MASK",
            "LD_ASSUME_KERNEL",
            "GLIBC_TUNABLES",
        ] {
            let effective = [(OsString::from(key), OsString::from("set"))]
                .into_iter()
                .collect();
            assert!(
                loader_state_is_ambiguous(&effective),
                "{key} can select a transitive object whose audit tags have not been inspected"
            );
        }
    }
}
