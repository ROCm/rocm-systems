//! Embed the rocjitsu configs mirage ships as agents, validating each
//! one against [`mirage_core::agent::AgentDef`] here rather than at run
//! time: the configs are source, not installed data — a released mirage
//! cannot go looking for them — and doing it here means a config that
//! stops parsing fails the build instead of every run.
//!
//! The whole directory is read, rather than a list kept here by hand.
//! That list was three entries long while `rocjitsu/configs` held eleven
//! usable ones, so mirage could emulate an MI210 or a W7900 and had no
//! name for either. Adding a GPU to rocjitsu now adds it to mirage.

// A build script may panic: a config that is missing, unparseable or no
// longer an agent is not something the crate can be built without, and
// there is no caller to return an error to.
#![allow(clippy::expect_used, clippy::panic)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// The names mirage addressed its first three agents by, before it read
/// the whole directory.
///
/// They are not derivable — `gfx1250_mi455x` describes an MI455X and
/// mirage has always called it `mi450x` — and they are load-bearing:
/// `--profile` defaults to `mi350x`, the shipped topologies name
/// `MI350X` and `MI300X`, and every agent and profile a user already has
/// on disk was written under them. Everything else is addressed by its
/// config's file stem, which is unique and needs no table.
const ESTABLISHED_NAMES: [(&str, &str); 3] = [
    ("gfx942_cdna3", "mi300x"),
    ("gfx950_mi355x", "mi350x"),
    ("gfx1250_mi455x", "mi450x"),
];

/// One config that could become an agent, before the pick below.
struct Candidate {
    stem: String,
    config: serde_json::Value,
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let configs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../rocjitsu/configs");
    // The directory as well as each file in it: a config that is *added*
    // changes no file mirage already reads, and without this the new GPU
    // would not appear until something else forced a rebuild.
    println!("cargo:rerun-if-changed={}", configs.display());

    let mut devices: BTreeMap<String, Vec<Candidate>> = BTreeMap::new();
    for path in json_files(&configs) {
        println!("cargo:rerun-if-changed={}", path.display());
        let stem = path
            .file_stem()
            .expect("a *.json path has a stem")
            .to_string_lossy()
            .into_owned();
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        let config: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|error| panic!("invalid JSON in {}: {error}", path.display()));
        // A DBT guest config (`guest_*.json`) describes which guest runs
        // on which host, not a machine — there is no agent in it to take.
        if !config["vm"].is_object() || !config["topology"].is_object() {
            continue;
        }
        devices
            .entry(device_key(&stem, &config))
            .or_default()
            .push(Candidate { stem, config });
    }

    let mut presets: BTreeMap<String, (String, serde_json::Value)> = BTreeMap::new();
    for (key, mut candidates) in devices {
        // One agent per device, so `mirage agent list` is a list of
        // machines rather than of files. rocjitsu ships up to three
        // configs for the same GPU — the standalone one, a `_kmd` one,
        // and an `_Ngpu` one — and the differences between them are not
        // differences mirage has to express: it synthesises its own
        // config for every session and writes `vm.gpu.num_gpus` from the
        // profile's `--gpus-per-node`, so an agent that bakes in a GPU
        // count would have that count overwritten. The plainest config
        // for the device is the one taken; `gfx90a_mi210_kmd` is the
        // MI210's only config and is therefore taken as it is.
        candidates.sort_by_key(|c| (variant_rank(&c.stem), c.stem.len(), c.stem.clone()));
        let chosen = candidates.remove(0);
        let name = ESTABLISHED_NAMES
            .iter()
            .find_map(|(stem, name)| (*stem == chosen.stem).then_some((*name).to_string()))
            .unwrap_or_else(|| chosen.stem.clone());
        check_name(&name, &chosen.stem);
        if let Some((taken, _)) = presets.get(&name) {
            panic!(
                "rocjitsu configs {taken}.json and {}.json both want the mirage name \
                 {name:?} (device key {key:?})",
                chosen.stem
            );
        }
        presets.insert(name, (chosen.stem, chosen.config));
    }

    // A rocjitsu config that is renamed or retired takes a mirage name
    // with it, and the three below are the ones nothing on a user's disk
    // survives losing. Fail here, where the fix is to add the new stem to
    // the table above, rather than shipping a mirage whose default
    // `--profile mi350x` names nothing.
    for (stem, name) in ESTABLISHED_NAMES {
        assert!(
            presets.get(name).is_some_and(|(chosen, _)| chosen == stem),
            "no rocjitsu config maps to the mirage name {name:?}: it came from \
             configs/{stem}.json, which is no longer the config chosen for its \
             device. Point ESTABLISHED_NAMES at the config that replaced it."
        );
    }

    let mut output = String::from("pub(crate) static PRESETS: &[Preset] = &[\n");
    for (name, (stem, config)) in &presets {
        let agent = serde_json::json!({"vm": config["vm"], "topology": config["topology"]});
        serde_json::from_value::<mirage_core::agent::AgentDef>(agent.clone())
            .unwrap_or_else(|error| panic!("invalid agent in {stem}.json: {error}"));
        let json = serde_json::to_string(&agent).expect("agent JSON is serializable");
        let device = &config["vm"]["gpu"]["device"];
        writeln!(
            output,
            "    Preset {{ name: {name:?}, stem: {stem:?}, marketing_name: {:?}, \
             arch: {:?}, agent: {json:?} }},",
            device["marketing_name"].as_str().unwrap_or_default(),
            config["vm"]["arch"].as_str().unwrap_or_default(),
        )
        .expect("writing to a String cannot fail");
    }
    output.push_str("];\n");

    let destination =
        PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR")).join("presets.rs");
    std::fs::write(&destination, output)
        .unwrap_or_else(|error| panic!("cannot write {}: {error}", destination.display()));
}

/// Every `*.json` in `dir`, in a fixed order.
///
/// Sorted, because `read_dir` is not: an unsorted walk would reorder the
/// generated table between builds on the same sources, and with it the
/// pick below whenever two candidates tie.
fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", dir.display()))
        .map(|entry| {
            entry
                .unwrap_or_else(|error| panic!("cannot read {}: {error}", dir.display()))
                .path()
        })
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    out.sort();
    out
}

/// What makes two configs the same machine.
///
/// `gfx_target_version` is the GPU's ISA version and is exactly the
/// grouping wanted: rocjitsu's `_kmd` and `_Ngpu` variants of a device
/// carry the same one, and no two devices share it. A config that does
/// not name it stands alone under its own stem rather than being pooled
/// with every other config that also omits it.
fn device_key(stem: &str, config: &serde_json::Value) -> String {
    match config["vm"]["gpu"]["device"]["gfx_target_version"].as_u64() {
        Some(version) => format!("gfx_target_version:{version}"),
        None => format!("stem:{stem}"),
    }
}

/// How far a config is from being the plain one for its device.
///
/// Lexical, from the naming convention `rocjitsu/docs/npi.md` lays down
/// (`configs/<gpu>.json` plus `_kmd` and any multi-GPU variant), and
/// deliberately so: it is the convention that makes `gfx950_mi355x` the
/// config for the MI350X and `gfx950_mi355x_kmd_2gpu` a variant of it,
/// and nothing inside the documents says which is which. A config
/// following no convention scores 0 and simply competes on its stem.
fn variant_rank(stem: &str) -> u32 {
    let mut rank = 0;
    for part in stem.split('_') {
        if part == "kmd" {
            rank += 1;
        }
        // `2gpu`, `4gpu`: a GPU count baked into a config mirage would
        // overwrite from the profile anyway, so these lose hardest.
        if let Some(count) = part.strip_suffix("gpu")
            && !count.is_empty()
            && count.chars().all(|c| c.is_ascii_digit())
        {
            rank += 2;
        }
    }
    rank
}

/// Refuse a config whose stem mirage could not address a document by.
///
/// Agents and profiles are stored lowercase under `<name>.json` and
/// restricted to `[A-Za-z0-9._+-]`, so a stem outside that set would
/// produce a builtin that no command could name. Caught here, against
/// the same rule the store enforces, rather than as a run-time refusal
/// on a document mirage itself wrote.
fn check_name(name: &str, stem: &str) {
    use mirage_core::store::{DocKind, validate_name};
    if let Err(error) = validate_name(DocKind::Agent, name) {
        panic!("rocjitsu config {stem}.json cannot be a mirage agent: {error}");
    }
    assert!(
        DocKind::Agent.canonical(name) == name,
        "rocjitsu config {stem}.json would be stored as \
         {:?} rather than {name:?}; give it a lowercase name in ESTABLISHED_NAMES",
        DocKind::Agent.canonical(name)
    );
}
