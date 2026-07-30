//! Turning a session description plus a command into a process grid.
//!
//! This is the single place that decides what a workload process actually
//! is: which program, in which container, with which environment, on
//! which streams. Both callers go through it —
//!
//! * `mirage run`, which describes the session it just brought up, and
//! * `mirage exec`, which was handed the same description over the run's
//!   socket
//!
//! — and that is the point. The two live in different processes and
//! different terminals, and if they built specs separately they would
//! drift: an exec would end up with a slightly different `LD_PRELOAD`, or
//! a different rendezvous, or the wrong workdir inside a container, and
//! the symptom would be a workload that runs correctly under `run` and
//! mysteriously fails under `exec`. Sharing the builder makes that class
//! of bug unrepresentable.
//!
//! Everything here is a pure function of its inputs, so the whole mapping
//! is testable without a container runtime or an emulator.

use std::collections::BTreeMap;

use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId};
use mirage_core::container::{
    ENV_HEAD_ADDR, ENV_HEAD_PORT, ENV_LOCAL_RANK, ENV_MASTER_ADDR, ENV_MASTER_PORT,
    ENV_NCCL_HOSTID, ENV_RANK, ENV_TORCH_RANK, ENV_WORLD_SIZE,
};
use mirage_core::proto::SessionDescription;

use crate::process::{ContainerProc, SpawnSpec, StdioMode};

/// Most processes one exec may start, across every node.
///
/// An emulated job is bounded by what the host can actually fork, and the
/// product of two user-supplied numbers is easy to get catastrophically
/// wrong: `--nodes 2 --nproc-per-node 2000000000` would otherwise try to
/// start four billion processes.
pub const MAX_WORLD_SIZE: u32 = 4096;

/// Per-session scratch directory inside every node container.
pub(crate) const CONTAINER_RUNTIME_DIR: &str = "/mnt/mirage/runtime";

/// Build the spawn specs for one exec.
///
/// `scratch` is the host directory the pid files land in; for a
/// containerised session it is the directory bind-mounted into every node
/// container, and this function creates the per-exec subdirectory under
/// it.
///
/// # Errors
///
/// Returns an error if the world size exceeds [`MAX_WORLD_SIZE`], or if
/// the pid-file directory cannot be created.
pub fn build_specs(
    desc: &SessionDescription,
    def: &ExecDef,
    exec_id: &ExecId,
) -> Result<Vec<SpawnSpec>> {
    let node_count = desc.node_count.max(1);
    let nproc = def.nproc_per_node.max(1);

    // The grid size is user-supplied, so the product has to be checked
    // rather than assumed: unchecked it panics in a debug build and wraps
    // in a release one, and either way the loop below starts forking.
    let world_size = node_count
        .checked_mul(nproc)
        .filter(|n| *n <= MAX_WORLD_SIZE)
        .ok_or_else(|| {
            MirageError::other(format!(
                "{node_count} nodes x {nproc} processes per node is {} processes, \
                 more than the {MAX_WORLD_SIZE} mirage will start for one exec",
                u64::from(node_count) * u64::from(nproc)
            ))
        })?;

    // Materialise the directory the containers write their pid files
    // into. The bind mount is read-write but the container cannot create
    // the path itself: `sh` would have to `mkdir -p` before redirecting,
    // and a redirect into a missing directory fails silently enough that
    // the first sign of trouble would be a signal that went nowhere.
    if let Some(targets) = &desc.containers {
        let dir = targets.scratch.join("exec").join(exec_id.as_str());
        std::fs::create_dir_all(&dir).map_err(|e| MirageError::io(dir, e))?;
    }

    let mut specs = Vec::with_capacity(world_size as usize);
    for node in 0..node_count {
        for local in 0..nproc {
            let global = node * nproc + local;
            let args = if node == 0 {
                &def.exec
            } else {
                def.worker_exec.as_ref().unwrap_or(&def.exec)
            };
            let env = process_env(desc, args, global, node, local, world_size);
            let stdio = StdioMode::for_rank(global, def.capture_all);

            specs.push(match &desc.containers {
                // Containerised: run the workload inside the node's
                // container via the provider. The provider client is what
                // we supervise; the workload lives in the container's PID
                // namespace and is signalled through the provider (see
                // [`ContainerProc`]).
                Some(targets) => {
                    let container = targets
                        .name(node)
                        .ok_or_else(|| {
                            MirageError::other(format!("session has no container for node {node}"))
                        })?
                        .to_string();
                    let engine = mirage_container::Engine::with_provider(&targets.provider);
                    let env_pairs: Vec<(String, String)> = env.into_iter().collect();
                    let pid_file = targets
                        .scratch
                        .join("exec")
                        .join(exec_id.as_str())
                        .join(format!("{global}.pid"));
                    let (command, rest) =
                        pid_recording_command(&args.command, &args.args, exec_id, global);
                    let argv = engine.exec_command_line(
                        &container,
                        args.workdir.as_deref(),
                        &env_pairs,
                        &command,
                        &rest,
                    );
                    let (command, rest) = argv
                        .split_first()
                        .map(|(c, r)| (c.clone(), r.to_vec()))
                        .unwrap_or_else(|| (targets.provider.clone(), Vec::new()));
                    SpawnSpec {
                        node: global,
                        command,
                        args: rest,
                        env: BTreeMap::new(),
                        workdir: None,
                        stdio,
                        // The provider CLI needs its own environment to
                        // locate its socket and configuration.
                        inherit_env: true,
                        container: Some(ContainerProc {
                            provider: targets.provider.clone(),
                            container,
                            pid_file,
                        }),
                    }
                }
                None => SpawnSpec {
                    node: global,
                    command: args.command.clone(),
                    args: args.args.clone(),
                    env,
                    // The session's working directory is a host path — the
                    // directory the caller was in — and is a sensible
                    // default for a process running on the host.
                    workdir: args.workdir.clone().or_else(|| Some(desc.workdir.clone())),
                    stdio,
                    // The caller's environment, unless they asked for a
                    // clean one. See [`SpawnSpec::inherit_env`].
                    inherit_env: !def.clear_env,
                    // The process we spawn *is* the workload, so
                    // signalling its group reaches it directly.
                    container: None,
                },
            });
        }
    }
    Ok(specs)
}

/// The environment one workload process runs with.
///
/// Layering order: the emulator's injection first, then the user's
/// per-exec environment, then mirage's own rank variables. Mirage's go
/// last so a workload cannot accidentally break its own rendezvous by
/// exporting `RANK` or `WORLD_SIZE`.
fn process_env(
    desc: &SessionDescription,
    args: &ExecArgs,
    global: u32,
    node: u32,
    local: u32,
    world_size: u32,
) -> BTreeMap<String, String> {
    let mut env = desc.env.clone();
    env.extend(args.env.clone());
    env.extend(proc_env(
        global,
        node,
        local,
        world_size,
        &desc.head_addr,
        desc.head_port,
    ));

    // `LD_PRELOAD` is the exception: the emulator's interposer and a
    // user-supplied preload must coexist, so they are concatenated rather
    // than one clobbering the other.
    if let Some(preload) = &desc.ld_preload {
        let combined = match args.env.get("LD_PRELOAD") {
            Some(user) if !user.is_empty() => format!("{preload}:{user}"),
            _ => preload.clone(),
        };
        env.insert("LD_PRELOAD".to_string(), combined);
    }
    env
}

/// The in-container path of a rank's pid file.
fn container_pid_file(exec: &ExecId, global: u32) -> String {
    format!("{CONTAINER_RUNTIME_DIR}/exec/{}/{global}.pid", exec.as_str())
}

/// Wrap a workload command so it records its own in-container pid before
/// becoming the workload.
///
/// Returns the command and arguments to hand the provider. The shell
/// writes `$$`, then `exec`s the real program *into that same pid*, so
/// the recorded number identifies the workload and not a wrapper that has
/// since exited. Without it there is no way to name the workload from
/// outside the container, and `podman exec` neither forwards signals to
/// it nor reports its pid.
///
/// The program and its arguments are passed as `$0`/`$@` rather than
/// interpolated into the script, so nothing about them is ever parsed by
/// the shell — a command containing a space, a quote or a `;` is handed
/// through byte-for-byte.
fn pid_recording_command(
    command: &str,
    args: &[String],
    exec: &ExecId,
    global: u32,
) -> (String, Vec<String>) {
    // The path is built from a constant and an `ExecId` (`e-` plus
    // digits), so it holds no shell metacharacters; it is quoted anyway.
    let pid_file = container_pid_file(exec, global);
    let script = format!("echo $$ > '{pid_file}' 2>/dev/null; exec \"$0\" \"$@\"");
    let mut argv = vec!["-c".to_string(), script, command.to_string()];
    argv.extend(args.iter().cloned());
    ("/bin/sh".to_string(), argv)
}

/// The mirage/`torch.distributed` environment for one workload process.
///
/// Three ranks identify a process: `node` (which emulated node it runs
/// on), `global` (its index across the whole job) and `local` (its index
/// within the node, which a workload typically uses to pin a GPU). With
/// the default of one process per node, `global == node` and `local == 0`.
fn proc_env(
    global: u32,
    node: u32,
    local: u32,
    world_size: u32,
    head_addr: &str,
    head_port: u16,
) -> Vec<(String, String)> {
    // Processes on the head node reach the rendezvous over loopback;
    // everyone else needs the head's address.
    let head = if node == 0 { "localhost" } else { head_addr };
    vec![
        (ENV_RANK.to_string(), node.to_string()),
        (ENV_TORCH_RANK.to_string(), global.to_string()),
        (ENV_HEAD_ADDR.to_string(), head.to_string()),
        (ENV_HEAD_PORT.to_string(), head_port.to_string()),
        (ENV_MASTER_ADDR.to_string(), head.to_string()),
        (ENV_MASTER_PORT.to_string(), head_port.to_string()),
        (ENV_WORLD_SIZE.to_string(), world_size.to_string()),
        (ENV_LOCAL_RANK.to_string(), local.to_string()),
        // Every emulated node runs on the same real host and is
        // synthesised from an identical config, so their GPUs report the
        // same location id and RCCL would reject them as duplicates. A
        // distinct host id per node is the correct model: one emulated
        // GPU per node.
        (ENV_NCCL_HOSTID.to_string(), format!("mirage-node-{node}")),
    ]
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use mirage_core::proto::ContainerTargets;
    use mirage_core::session::SessionId;

    fn desc(node_count: u32) -> SessionDescription {
        SessionDescription {
            session: SessionId::new("s").unwrap(),
            node_count,
            workdir: "/work".to_string(),
            containers: None,
            env: BTreeMap::from([("EMU".to_string(), "1".to_string())]),
            ld_preload: Some("/lib/interpose.so".to_string()),
            head_addr: "mirage-s-node-0".to_string(),
            head_port: 6000,
        }
    }

    fn exec_def(nproc: u32, capture_all: bool) -> ExecDef {
        ExecDef {
            timestamp: chrono::Utc::now(),
            session: SessionId::new("s").unwrap(),
            exec: ExecArgs {
                command: "/bin/true".to_string(),
                args: vec![],
                env: BTreeMap::new(),
                workdir: None,
            },
            worker_exec: None,
            nproc_per_node: nproc,
            capture_all,
            clear_env: false,
        }
    }

    fn id() -> ExecId {
        ExecId::new("e-1").unwrap()
    }

    #[test]
    fn only_rank_zero_inherits_stdin() {
        // Two processes cannot both read one terminal; giving stdin to
        // more than one rank means keystrokes go to whichever happens to
        // read first.
        let specs = build_specs(&desc(3), &exec_def(1, false), &id()).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].stdio, StdioMode::Inherit { stdin: true });
        assert_eq!(specs[1].stdio, StdioMode::Inherit { stdin: false });
        assert_eq!(specs[2].stdio, StdioMode::Inherit { stdin: false });
    }

    #[test]
    fn capture_all_captures_every_rank_and_gives_none_of_them_stdin() {
        let specs = build_specs(&desc(2), &exec_def(1, true), &id()).unwrap();
        assert!(specs.iter().all(|s| s.stdio == StdioMode::Capture));
    }

    #[test]
    fn the_emulator_interposer_is_prepended_to_a_user_preload() {
        let mut def = exec_def(1, false);
        def.exec
            .env
            .insert("LD_PRELOAD".to_string(), "/user/mine.so".to_string());
        let specs = build_specs(&desc(1), &def, &id()).unwrap();
        assert_eq!(
            specs[0].env.get("LD_PRELOAD").map(String::as_str),
            Some("/lib/interpose.so:/user/mine.so")
        );
    }

    #[test]
    fn rank_variables_win_over_a_user_supplied_rank() {
        // A workload exporting RANK would otherwise break its own
        // rendezvous.
        let mut def = exec_def(1, false);
        def.exec
            .env
            .insert("RANK".to_string(), "99".to_string());
        let specs = build_specs(&desc(2), &def, &id()).unwrap();
        assert_eq!(specs[1].env.get("RANK").map(String::as_str), Some("1"));
    }

    #[test]
    fn local_and_global_ranks_are_distinct_with_several_procs_per_node() {
        let specs = build_specs(&desc(2), &exec_def(2, false), &id()).unwrap();
        assert_eq!(specs.len(), 4);
        let at = |i: usize, k: &str| specs[i].env.get(k).cloned().unwrap_or_default();
        assert_eq!(at(3, "RANK"), "3");
        assert_eq!(at(3, "LOCAL_RANK"), "1");
        assert_eq!(at(3, "MIRAGE_RANK"), "1");
        assert_eq!(at(3, "WORLD_SIZE"), "4");
    }

    #[test]
    fn the_head_node_uses_loopback_and_workers_use_the_head_address() {
        let specs = build_specs(&desc(2), &exec_def(1, false), &id()).unwrap();
        assert_eq!(
            specs[0].env.get("MASTER_ADDR").map(String::as_str),
            Some("localhost")
        );
        assert_eq!(
            specs[1].env.get("MASTER_ADDR").map(String::as_str),
            Some("mirage-s-node-0")
        );
    }

    #[test]
    fn an_impossible_world_size_is_refused_rather_than_forked() {
        let err = build_specs(&desc(2), &exec_def(u32::MAX, false), &id()).unwrap_err();
        assert!(err.to_string().contains("more than the"), "{err}");
    }

    #[test]
    fn a_containerised_exec_goes_through_the_provider_and_records_its_pid() {
        let scratch = tempfile::tempdir().unwrap();
        let mut d = desc(2);
        d.containers = Some(ContainerTargets {
            provider: "podman".to_string(),
            names: vec!["mirage-s-node-0".to_string(), "mirage-s-node-1".to_string()],
            scratch: scratch.path().to_path_buf(),
        });
        let specs = build_specs(&d, &exec_def(1, false), &id()).unwrap();

        assert_eq!(specs[1].command, "podman");
        assert_eq!(specs[1].args[0], "exec");
        // The workload is wrapped so it can be signalled from outside the
        // container's PID namespace.
        assert!(specs[1].args.iter().any(|a| a == "/bin/sh"));
        assert!(specs[1].args.iter().any(|a| a.contains("echo $$")));
        assert_eq!(
            specs[1].container.as_ref().map(|c| c.container.as_str()),
            Some("mirage-s-node-1")
        );
        // The directory the wrapper redirects into must exist before the
        // container tries to write there.
        assert!(scratch.path().join("exec").join("e-1").is_dir());
    }

    #[test]
    fn a_workload_inherits_the_callers_environment_by_default() {
        // Mirage's parent is the terminal the user typed in, so what they
        // exported there — a token, a PYTHONPATH, a proxy — they meant
        // for the workload.
        let specs = build_specs(&desc(1), &exec_def(1, false), &id()).unwrap();
        assert!(specs[0].inherit_env);
    }

    #[test]
    fn clear_env_asks_for_an_almost_empty_environment() {
        let mut def = exec_def(1, false);
        def.clear_env = true;
        let specs = build_specs(&desc(1), &def, &id()).unwrap();
        assert!(!specs[0].inherit_env);
        // The emulator's own variables survive either way: they are
        // layered on explicitly, not inherited, and a workload that lost
        // them would run unemulated.
        assert_eq!(specs[0].env.get("EMU").map(String::as_str), Some("1"));
    }

    #[test]
    fn clearing_the_environment_does_not_change_a_containerised_exec() {
        // A container never inherits the host environment; the workload
        // sees exactly what mirage passes with `-e`. The flag governs the
        // provider *client*, which needs its own environment to find its
        // socket whatever the user asked for.
        let scratch = tempfile::tempdir().unwrap();
        let mut d = desc(1);
        d.containers = Some(ContainerTargets {
            provider: "podman".to_string(),
            names: vec!["mirage-s-node-0".to_string()],
            scratch: scratch.path().to_path_buf(),
        });
        let mut def = exec_def(1, false);
        def.clear_env = true;
        let specs = build_specs(&d, &def, &id()).unwrap();
        assert!(specs[0].inherit_env);
    }

    #[test]
    fn a_command_with_shell_metacharacters_is_not_parsed_by_the_wrapper() {
        let (command, argv) = pid_recording_command(
            "/bin/echo",
            &["a b; rm -rf /".to_string()],
            &id(),
            0,
        );
        assert_eq!(command, "/bin/sh");
        // The dangerous argument is passed positionally, never spliced
        // into the script text.
        assert_eq!(argv.last().map(String::as_str), Some("a b; rm -rf /"));
        assert!(!argv[1].contains("rm -rf"), "{:?}", argv[1]);
    }
}
