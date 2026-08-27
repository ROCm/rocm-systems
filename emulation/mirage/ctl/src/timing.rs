//! Assembling the timing numbers one `mirage run` bakes into its config.
//!
//! Three layers, in this order: the device's own built-in table, the
//! overrides a profile stores, and a tuning file named on the command
//! line. Each may only replace a key the layer below already has, and
//! all three are reported at start-up, because a number nobody can
//! account for reads exactly like one that was measured.
//!
//! What mirage produces is a self-contained config with the numbers in
//! it, not a config that points at them. A run's timing is then
//! reproducible from the one artefact it was handed, and that artefact
//! can be given to somebody else without them needing anything more.
//!
//! A tuning file a user supplies is read and merged and nothing else:
//! it is never copied into mirage's configuration directory or into this
//! repository. The merged result is written only into the session's
//! scratch directory under `$XDG_RUNTIME_DIR`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use mirage_core::common::{MaybeRef, SimpleMap};
use mirage_core::profile::ProfileDef;
use mirage_core::timing::{
    TimingTable, enabled_in_options, overrides_from_options, read_tuning_document,
    unknown_keys_message, write_options,
};
use serde_json::Number;

use crate::RunArgs;

/// Keys named in a report line before it stops naming them.
///
/// A tuning file may replace half the table, and a start-up line that
/// prints forty keys is one nobody reads — including the reader who
/// needed to notice the one key they did not mean to override.
const KEYS_SHOWN: usize = 6;

/// What this run's timing will be, and where each part of it came from.
#[derive(Debug, Default)]
pub(crate) struct Plan {
    /// Whether a timing block will be baked at all.
    enabled: bool,
    /// The device's shader clock, for the report.
    clock_mhz: Option<u64>,
    /// How the agent was named, for the report.
    agent: String,
    /// How many numbers the device's own table has.
    builtin_keys: usize,
    /// Keys the profile replaces, sorted.
    from_profile: Vec<String>,
    /// The tuning file, and the keys it replaces.
    tuning_file: Option<PathBuf>,
    from_file: Vec<String>,
    /// The two override layers, folded.
    merged: BTreeMap<String, Number>,
}

impl Plan {
    /// Record this plan in a profile's emulator options.
    ///
    /// Only called when the run actually said something about timing;
    /// a run that says nothing leaves the profile's own selection alone,
    /// which is what keeps the cheap by-name profile reference.
    pub(crate) fn apply(&self, options: &mut SimpleMap) {
        write_options(options, self.enabled, &self.merged);
    }

    /// What `mirage run` says about timing before it starts anything.
    ///
    /// Empty when the run neither asked for timing nor turned it off: a
    /// functional run has nothing to report, and a line saying so on
    /// every run is a line nobody reads on the run where it matters.
    pub(crate) fn report_lines(&self, asked_off: bool) -> Vec<String> {
        if !self.enabled {
            return if asked_off {
                vec!["timing off; the emulator will not model device time".to_string()]
            } else {
                Vec::new()
            };
        }
        let clock = self.clock_mhz.map_or_else(
            || "an unstated clock".to_string(),
            |mhz| format!("{mhz} MHz"),
        );
        let mut out = vec![format!(
            "timing on at {clock}: {} numbers baked in from {}",
            self.builtin_keys, self.agent
        )];
        if self.from_profile.is_empty() && self.from_file.is_empty() {
            out.push("timing: none of them overridden".to_string());
            return out;
        }
        if !self.from_profile.is_empty() {
            out.push(format!(
                "timing: {} replaced by the profile ({})",
                self.from_profile.len(),
                shown(&self.from_profile)
            ));
        }
        if let Some(path) = &self.tuning_file {
            out.push(format!(
                "timing: {} replaced by {} ({})",
                self.from_file.len(),
                path.display(),
                shown(&self.from_file)
            ));
        }
        out
    }
}

/// The first few keys of a list, and how many were not named.
fn shown(keys: &[String]) -> String {
    if keys.len() <= KEYS_SHOWN {
        return keys.join(", ");
    }
    format!(
        "{}, and {} more",
        keys[..KEYS_SHOWN].join(", "),
        keys.len() - KEYS_SHOWN
    )
}

/// Work out what this run's timing will be, refusing anything that
/// cannot be baked.
///
/// The device's table is resolved only when timing is actually wanted.
/// A functional run must not start failing because a profile's agent
/// reference is broken in a way that only a timing table would have
/// noticed.
///
/// # Errors
///
/// Returns an error when the tuning file cannot be read or parsed, when
/// the agent carries no timing table, or when either override layer
/// names a key the device does not have.
pub(crate) fn plan(profile: &ProfileDef, a: &RunArgs) -> anyhow::Result<Plan> {
    // A drop-in config is the whole description of the machine, timing
    // included, so mirage has nothing to add to it. The flags that would
    // have added something are refused by clap; a profile that stores
    // both is simply told the config wins, by this returning nothing.
    let drop_in = a.config.is_some() || profile.emulator.options.contains_key("config");
    let enabled = !a.no_timing
        && !drop_in
        && (a.timing || a.timing_tuning.is_some() || enabled_in_options(&profile.emulator.options));
    if !enabled {
        return Ok(Plan::default());
    }

    // A backend that has no device time to model must not be reported as
    // modelling it. Only checked when this build actually has the backend
    // compiled in: with no entry there is no schema to check against, and
    // refusing on that basis would blame the flag for a missing backend,
    // exactly as `-o` and `--plugin` decline to.
    if let Some(spec) = crate::find_emulator(&profile.emulator.emulator)
        && !spec.models_time
    {
        anyhow::bail!(
            "emulator `{}` has no device time to model, so it cannot be timed. \
             A backend that runs the workload on real hardware already has the \
             hardware's timing, and one that only translates has somebody else's. \
             `mirage emulators -l` lists what is here.",
            spec.name
        );
    }

    let (agent_name, stored_as, table) = device_table(profile)?;
    if table.is_empty() {
        // `mirage state builtins` will not do this on its own: an agent
        // from before mirage shipped timing tables differs from the one
        // mirage now ships, which is indistinguishable from an agent the
        // user edited, and it leaves those alone by design. Deleting a
        // builtin puts the shipped one back in the same breath, which is
        // why that is the command named here.
        let fix = stored_as.map_or_else(
            || "add a `timing` object to it".to_string(),
            |name| {
                format!(
                    "run `mirage agent delete {name}`, which puts the shipped agent — \
                     timing table and all — straight back in its place"
                )
            },
        );
        anyhow::bail!(
            "timing was asked for and {agent_name} carries no timing table. An agent \
             from before mirage shipped one keeps working functionally; to time it, {fix}."
        );
    }

    // The profile's own overrides first, then the file's, so that the
    // file wins and each layer is blamed by name when it is the one with
    // the bad key in it.
    let mut resolved = table.clone();
    let from_profile =
        overrides_from_options(&profile.emulator.options).map_err(|e| anyhow::anyhow!("{e}"))?;
    overlay(&mut resolved, &from_profile, &table, "this profile")?;

    let (tuning_file, from_file) = match &a.timing_tuning {
        Some(path) => {
            let (abs, overrides) = read_tuning_file(path)?;
            overlay(
                &mut resolved,
                &overrides,
                &table,
                &abs.display().to_string(),
            )?;
            (Some(abs), overrides)
        }
        None => (None, BTreeMap::new()),
    };

    let mut merged = from_profile.clone();
    merged.extend(from_file.iter().map(|(k, v)| (k.clone(), v.clone())));
    Ok(Plan {
        enabled: true,
        clock_mhz: resolved
            .get(mirage_core::timing::CLOCK_KEY)
            .and_then(Number::as_u64),
        agent: agent_name,
        builtin_keys: table.len(),
        from_profile: from_profile.into_keys().collect(),
        tuning_file,
        from_file: from_file.into_keys().collect(),
        merged,
    })
}

/// Apply one override layer, naming the layer when a key is refused.
fn overlay(
    resolved: &mut TimingTable,
    overrides: &BTreeMap<String, Number>,
    table: &TimingTable,
    source: &str,
) -> anyhow::Result<()> {
    resolved
        .overlay(overrides)
        .map_err(|unknown| anyhow::anyhow!(unknown_keys_message(&unknown, table, source)))
}

/// The timing table of the device this profile emulates, and how to name
/// it in a message.
fn device_table(profile: &ProfileDef) -> anyhow::Result<(String, Option<String>, TimingTable)> {
    let topology = match &profile.emulator.topology {
        MaybeRef::Owned(t) => t.clone(),
        // Through the store's front door, so the name is checked before
        // it is joined to the config directory; see the same resolution
        // in `apply_profile_overrides`.
        MaybeRef::Ref(name) => mirage_core::store::topology_get(name)?,
    };
    Ok(match &topology.agent {
        MaybeRef::Owned(agent) => (
            "the profile's inline agent".to_string(),
            None,
            agent.timing.clone(),
        ),
        MaybeRef::Ref(name) => {
            let agent = mirage_core::store::agent_get(name)?;
            (format!("agent `{name}`"), Some(name.clone()), agent.timing)
        }
    })
}

/// Read a `--timing-tuning` file into dotted keys.
///
/// The path is made absolute for the report and for the error, so that
/// two files of the same name in different directories are
/// distinguishable in a log somebody reads a week later.
///
/// The contents are parsed here, unlike a drop-in `--config`, because
/// these keys are checked against the device's table and mirage is the
/// only party that has both in hand. What it does *not* do is keep the
/// file: it is read, merged, and left where the user put it.
fn read_tuning_file(path: &str) -> anyhow::Result<(PathBuf, BTreeMap<String, Number>)> {
    let abs = std::fs::canonicalize(path).map_err(|e| {
        anyhow::anyhow!("--timing-tuning {path}: {e}. Name a tuning file that exists.")
    })?;
    if abs.is_dir() {
        anyhow::bail!("--timing-tuning {path}: this is a directory, not a tuning file.");
    }
    let text = std::fs::read_to_string(&abs)
        .map_err(|e| anyhow::anyhow!("--timing-tuning {}: {e}", abs.display()))?;
    let document: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("--timing-tuning {}: not JSON: {e}", abs.display()))?;
    let overrides =
        read_tuning_document(&document).map_err(|e| anyhow::anyhow!("{}: {e}", abs.display()))?;
    Ok((abs, overrides))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use mirage_core::agent::AgentDef;
    use mirage_core::topology::TopologyDef;

    /// A backend that has no device time cannot be asked for it, and is
    /// told so rather than reported as timing a run it will not time.
    #[test]
    fn a_backend_with_no_device_time_is_refused() {
        let mut p = profile_with(mirage_builtin::mi350x());
        p.emulator.emulator = "hotswap".to_string();
        let refused = plan(&p, &args(&["--timing"]));
        // This build may not have the backend compiled in, in which case
        // there is nothing to check against and nothing to refuse.
        if crate::find_emulator("hotswap").is_some() {
            let e = refused.unwrap_err().to_string();
            assert!(e.contains("no device time"), "{e}");
        } else {
            assert!(refused.is_ok());
        }
    }

    /// A profile whose agent is inline, so no store and no filesystem is
    /// involved in resolving it.
    fn profile_with(agent: AgentDef) -> ProfileDef {
        let mut p = crate::tests::sample_profile();
        p.emulator.topology = MaybeRef::Owned(TopologyDef {
            num_nodes: 1,
            gpus_per_node: 1,
            agent: MaybeRef::Owned(agent),
        });
        p
    }

    fn args(flags: &[&str]) -> RunArgs {
        let mut argv = flags.to_vec();
        argv.extend_from_slice(&["--", "./app"]);
        crate::tests::parse_run(&argv).expect("these flags should parse")
    }

    fn tuning_file(dir: &std::path::Path, body: &str) -> String {
        let path = dir.join("tuning.json");
        std::fs::write(&path, body).unwrap();
        path.display().to_string()
    }

    #[test]
    fn a_run_that_says_nothing_about_timing_bakes_nothing() {
        let p = profile_with(mirage_builtin::mi350x());
        let plan = plan(&p, &args(&[])).unwrap();
        assert!(!plan.enabled);
        assert!(plan.report_lines(false).is_empty());
    }

    #[test]
    fn timing_bakes_the_agents_own_table() {
        let agent = mirage_builtin::mi350x();
        let want = agent.timing.len();
        let plan = plan(&profile_with(agent), &args(&["--timing"])).unwrap();
        assert!(plan.enabled);
        assert_eq!(plan.builtin_keys, want);
        assert_eq!(plan.clock_mhz, Some(2700));
        assert!(plan.merged.is_empty(), "nothing was overridden");
        let report = plan.report_lines(false).join("\n");
        assert!(report.contains("2700 MHz"), "{report}");
        assert!(report.contains("none of them overridden"), "{report}");
    }

    /// The three layers, applied in the order the docs promise: the file
    /// beats the profile, the profile beats the agent, and every layer is
    /// named in the report.
    #[test]
    fn the_file_beats_the_profile_which_beats_the_agent() {
        let tmp = tempfile::tempdir().unwrap();
        let mut p = profile_with(mirage_builtin::mi350x());
        p.emulator.options.insert(
            "timing.machine.xcds".to_string(),
            mirage_core::common::SimpleValue::String("4".to_string()),
        );
        p.emulator.options.insert(
            "timing.machine.lds.banks".to_string(),
            mirage_core::common::SimpleValue::String("16".to_string()),
        );
        let file = tuning_file(
            tmp.path(),
            r#"{"xcds": 2, "dram": {"latency_cycles": 900}}"#,
        );

        let plan = plan(&p, &args(&["--timing-tuning", &file])).unwrap();
        assert_eq!(plan.merged["xcds"], Number::from(2), "the file wins");
        assert_eq!(plan.merged["lds.banks"], Number::from(16));
        assert_eq!(plan.merged["dram.latency_cycles"], Number::from(900));

        let report = plan.report_lines(false).join("\n");
        assert!(report.contains("2 replaced by the profile"), "{report}");
        assert!(report.contains("2 replaced by "), "{report}");
        assert!(report.contains("tuning.json"), "{report}");
    }

    /// `--timing-tuning` on its own turns timing on: it is only ever
    /// typed by somebody who wants those numbers used, and reading a
    /// tuning file into a functional run would be a silent no-op.
    #[test]
    fn a_tuning_file_alone_turns_timing_on() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tuning_file(tmp.path(), r#"{"xcds": 2}"#);
        let plan = plan(
            &profile_with(mirage_builtin::mi350x()),
            &args(&["--timing-tuning", &file]),
        )
        .unwrap();
        assert!(plan.enabled);
    }

    #[test]
    fn an_unknown_key_is_an_error_and_not_a_new_parameter() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tuning_file(tmp.path(), r#"{"dram.latency_cyles": 900}"#);
        let e = plan(
            &profile_with(mirage_builtin::mi350x()),
            &args(&["--timing-tuning", &file]),
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("dram.latency_cyles"), "{e}");
        // And it says what the right spelling would have been.
        assert!(e.contains("dram.latency_cycles"), "{e}");
    }

    #[test]
    fn a_profile_override_of_an_unknown_key_is_refused_too() {
        let mut p = profile_with(mirage_builtin::mi350x());
        p.emulator.options.insert(
            "timing.machine.nonsense".to_string(),
            mirage_core::common::SimpleValue::String("1".to_string()),
        );
        let e = plan(&p, &args(&["--timing"])).unwrap_err().to_string();
        assert!(e.contains("this profile"), "{e}");
        assert!(e.contains("nonsense"), "{e}");
    }

    #[test]
    fn no_timing_beats_a_profile_that_asks_for_it() {
        let mut p = profile_with(mirage_builtin::mi350x());
        p.emulator.options.insert(
            "timing".to_string(),
            mirage_core::common::SimpleValue::Boolean(true),
        );
        p.emulator.options.insert(
            "timing.machine.xcds".to_string(),
            mirage_core::common::SimpleValue::String("4".to_string()),
        );
        let plan = plan(&p, &args(&["--no-timing"])).unwrap();
        assert!(!plan.enabled);

        // And applying it clears both halves, or a later run that turns
        // timing back on inherits an override nobody reported.
        let mut options = p.emulator.options.clone();
        plan.apply(&mut options);
        assert!(!options.contains_key("timing"));
        assert!(!options.keys().any(|k| k.starts_with("timing.machine.")));
        assert_eq!(
            plan.report_lines(true),
            vec!["timing off; the emulator will not model device time".to_string()]
        );
    }

    /// A profile that enables timing itself is honoured by a run that
    /// says nothing, which is what makes `--no-timing` necessary.
    #[test]
    fn a_profiles_own_selection_survives_a_silent_run() {
        let mut p = profile_with(mirage_builtin::mi350x());
        p.emulator.options.insert(
            "timing".to_string(),
            mirage_core::common::SimpleValue::Boolean(true),
        );
        assert!(plan(&p, &args(&[])).unwrap().enabled);
    }

    #[test]
    fn a_missing_or_unparseable_tuning_file_is_refused_by_name() {
        let p = profile_with(mirage_builtin::mi350x());
        let e = plan(&p, &args(&["--timing-tuning", "/nonexistent/t.json"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("/nonexistent/t.json"), "{e}");

        let tmp = tempfile::tempdir().unwrap();
        let bad = tuning_file(tmp.path(), "not json at all");
        let e = plan(&p, &args(&["--timing-tuning", &bad]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("not JSON"), "{e}");
    }

    /// An agent from before mirage had a timing table is refused loudly
    /// rather than timed on fallbacks, which would read slow and look
    /// like a measurement of a slow machine.
    #[test]
    fn an_agent_with_no_table_cannot_be_timed() {
        let e = plan(&profile_with(AgentDef::default()), &args(&["--timing"]))
            .unwrap_err()
            .to_string();
        assert!(e.contains("no timing table"), "{e}");
        // An inline agent has no name to delete, so it is told the other
        // half of the answer instead.
        assert!(e.contains("add a `timing` object"), "{e}");
    }

    /// A drop-in config is the whole machine description, timing
    /// included. The flags are refused by clap; a profile that stores
    /// both must not have a block added behind the config's back.
    #[test]
    fn a_drop_in_config_is_left_entirely_alone() {
        let mut p = profile_with(mirage_builtin::mi350x());
        p.emulator.options.insert(
            "timing".to_string(),
            mirage_core::common::SimpleValue::Boolean(true),
        );
        p.emulator.options.insert(
            "config".to_string(),
            mirage_core::common::SimpleValue::String("/tmp/cfg.json".to_string()),
        );
        assert!(!plan(&p, &args(&[])).unwrap().enabled);
    }

    /// The emitted config is itself a legal tuning document, so a baked
    /// config can be edited and handed back with `--timing-tuning`.
    #[test]
    fn a_baked_block_can_be_fed_back_in() {
        let tmp = tempfile::tempdir().unwrap();
        let block = mirage_builtin::mi350x().timing.to_config_block().unwrap();
        let file = tuning_file(tmp.path(), &block["machine"].to_string());
        let plan = plan(
            &profile_with(mirage_builtin::mi350x()),
            &args(&["--timing-tuning", &file]),
        )
        .unwrap();
        assert!(plan.enabled);
        assert_eq!(plan.merged.len(), mirage_builtin::mi350x().timing.len() - 1);
    }

    #[test]
    fn a_long_override_list_is_summarised() {
        let keys: Vec<String> = (0..10).map(|i| format!("k{i}")).collect();
        let line = shown(&keys);
        assert!(line.contains("and 4 more"), "{line}");
    }
}
