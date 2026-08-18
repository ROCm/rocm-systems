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
//! The mapping is very nearly a function of its inputs, and everything
//! it decides is testable without a container runtime or an emulator.
//! The one thing it asks the world about is a containerised
//! `--workdir`, which is a question only the image can answer and is
//! asked only when the caller passed one. Two more things are read from
//! the process rather than passed in, and both are properties
//! of the terminal mirage was started from rather than of the session:
//! whether the caller's streams are a terminal, and which runtime
//! directory this mirage resolved. Both callers — `mirage run` and
//! `mirage exec` — are the process the user is sitting in front of, which
//! is what makes reading them here correct.

use std::collections::BTreeMap;

use mirage_core::container::{
    ENV_HEAD_ADDR, ENV_HEAD_PORT, ENV_LOCAL_RANK, ENV_MASTER_ADDR, ENV_MASTER_PORT,
    ENV_NCCL_HOSTID, ENV_RANK, ENV_RUNTIME, ENV_SESSION, ENV_TORCH_RANK, ENV_WORLD_SIZE,
};
use mirage_core::error::{MirageError, Result};
use mirage_core::exec::{ExecArgs, ExecDef, ExecId};
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
/// # Identity: the session's grid, not this exec's
///
/// Two counts meet here and they are not the same number.
///
/// [`SessionDescription::nproc_per_node`] is the **job's** shape — what
/// the owning `mirage run` was started with — and it is what every rank
/// variable is computed against: `WORLD_SIZE` is `node_count *
/// nproc_per_node`, and a process is rank `node * nproc_per_node + local`
/// of it. [`ExecDef::nproc_per_node`] is how many of a node's slots
/// *this* invocation fills, which is a question about how many processes
/// to start and about nothing else.
///
/// For `mirage run` the two are equal, which is why conflating them went
/// unnoticed: the run's own exec is the job. For `mirage exec` they are
/// not. Computing the world from the exec's count told a
/// `mirage exec` into a `--nproc-per-node 3` job that the world had two
/// processes in it rather than six, and numbered its ranks accordingly —
/// while still pointing them at the run's rendezvous port, so a
/// collective built on those numbers mis-forms instead of failing.
///
/// The rule that falls out, and the one the CLI documents:
///
/// * `mirage exec -- cmd` starts one process per node, each of them its
///   node's local rank 0 — ranks `0`, `P`, `2P`, … of a `WORLD_SIZE` of
///   `N * P`.
/// * `mirage exec --node n -- cmd` starts exactly that node's local rank
///   0, which is what keeps it a single-process exec and therefore an
///   interactive one.
/// * `--nproc-per-node k` fills the first `k` slots of each node it runs
///   on, and `k` may not exceed the job's own `P`: rank `node * P + P` is
///   the next node's rank 0, and two live processes claiming one rank in
///   one rendezvous is the failure this whole section exists to prevent.
///
/// # Where a containerised workload starts
///
/// The two paths do not share a default working directory, and cannot.
///
/// On the host, a workload with no `--workdir` starts in
/// [`SessionDescription::workdir`] — the directory `mirage run` was
/// typed in — which is what makes `mirage run -- ./my-app` behave like
/// running `./my-app`.
///
/// In a container there is no equivalent, because the host's directory
/// is a host path and the container has its own filesystem. Carrying it
/// across would fail every run started anywhere but a bind-mounted
/// directory, and fail it in the most confusing possible way: with a
/// `chdir` error about a path the user never asked for. So mirage passes
/// no `-w` at all and the workload starts wherever the image says it
/// does — its `WORKDIR`, which is `/` for an image that declares none.
/// That is the image's answer to the same question rather than a guess
/// of mirage's, and `--workdir` overrides it; a path there is checked
/// against the container before anything runs, so getting it wrong costs
/// an error and not a failed workload.
///
/// # Errors
///
/// Returns an error if the world size exceeds [`MAX_WORLD_SIZE`], if the
/// exec asks for more processes per node than the job has slots for, if
/// a containerised `--workdir` names a directory the image does not
/// have, or if the pid-file directory cannot be created.
pub fn build_specs(
    desc: &SessionDescription,
    def: &ExecDef,
    exec_id: &ExecId,
) -> Result<Vec<SpawnSpec>> {
    let node_count = desc.node_count.max(1);
    // The job's shape and this exec's, kept apart deliberately; see the
    // "Identity" section above for what conflating them cost.
    let job_nproc = desc.nproc_per_node.max(1);
    let nproc = def.nproc_per_node.max(1);

    // The grid size is user-supplied, so the product has to be checked
    // rather than assumed: unchecked it panics in a debug build and wraps
    // in a release one, and either way the loop below starts forking.
    let world_size = node_count
        .checked_mul(job_nproc)
        .filter(|n| *n <= MAX_WORLD_SIZE)
        .ok_or_else(|| {
            MirageError::other(format!(
                "{node_count} nodes x {job_nproc} processes per node is {} processes, \
                 more than the {MAX_WORLD_SIZE} mirage will start for one exec",
                u64::from(node_count) * u64::from(job_nproc)
            ))
        })?;

    if nproc > job_nproc {
        return Err(MirageError::other(format!(
            "this session runs {job_nproc} process(es) per node, so it has no \
             {nproc}th slot on a node to start one in. Start the run with \
             `--nproc-per-node {nproc}` if that is the shape the job should have."
        )));
    }

    // Which nodes this exec actually starts processes on. `--node N`
    // narrows it to one; the rank variables below are still computed
    // against the *session's* full size, so a process started that way
    // sees exactly what its neighbours see.
    let nodes: Vec<u32> = match def.node {
        Some(node) if node >= node_count => {
            return Err(MirageError::other(format!(
                "this session has {node_count} node(s), so there is no node {node} \
                 (they are numbered 0..{})",
                node_count - 1
            )));
        }
        Some(node) => vec![node],
        None => (0..node_count).collect(),
    };

    // Materialise the directory the containers write their pid files
    // into. The bind mount is read-write but the container cannot create
    // the path itself: `sh` would have to `mkdir -p` before redirecting,
    // and a redirect into a missing directory fails silently enough that
    // the first sign of trouble would be a signal that went nowhere.
    if let Some(targets) = &desc.containers {
        let dir = targets.scratch.join("exec").join(exec_id.as_str());
        std::fs::create_dir_all(&dir).map_err(|e| MirageError::io(dir, e))?;
    }

    // One process gets the terminal; several share none of it. Decided
    // once, from the size of *this* exec, so every rank agrees — see
    // [`StdioMode::for_exec`].
    let stdio = StdioMode::for_exec(nodes.len() * nproc as usize);

    // And, for a containerised exec, whether to ask the provider for a
    // pseudo-terminal inside the container. Probed here rather than
    // deeper down because this is the process the user is sitting in
    // front of: both callers of `build_specs` — `mirage run` and
    // `mirage exec` — own the terminal their workload will run on.
    let tty = {
        use std::io::IsTerminal as _;
        wants_tty(
            stdio,
            std::io::stdin().is_terminal()
                && std::io::stdout().is_terminal()
                && std::io::stderr().is_terminal(),
        )
    };

    // The runtime directory that owns everything this call starts, read
    // from the environment rather than from `desc` because both callers
    // necessarily agree on it: a `mirage exec` reached this session
    // through a socket in that very directory, so a client that resolved
    // a different one would not have found the run at all.
    //
    // Resolved once. It is the same for every rank, and canonicalising it
    // per process would put a syscall inside the loop for no answer that
    // could differ.
    let runtime = mirage_core::container::owning_runtime();

    // Which `--workdir` values have already been put to a container.
    // Every node of a session runs the same image with the same mounts,
    // so one answer settles the question for all of them, and an exec
    // that names a worker command has at most two distinct directories
    // between them. Without this the probe would be an extra provider
    // invocation per rank of a wide job, to be told the same thing each
    // time.
    let mut asked: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();

    let mut specs = Vec::with_capacity(nodes.len() * nproc as usize);
    for node in nodes {
        for local in 0..nproc {
            // The job's stride, not this exec's: rank `n` of the session
            // is at `n * job_nproc`, whoever started it.
            let global = node * job_nproc + local;
            let args = if node == 0 {
                &def.exec
            } else {
                def.worker_exec.as_ref().unwrap_or(&def.exec)
            };
            let env = process_env(desc, args, global, node, local, world_size, &runtime);

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
                    // No [`Cancel`](mirage_container::Cancel) on this
                    // one. It builds an argv, and asks the container one
                    // bounded question about its filesystem below;
                    // neither is a step a user waits on, so there is
                    // nothing here for an interrupt to shorten. The
                    // engine that does run long providers — the
                    // session's bring-up — is given the session's switch
                    // where it is built, in
                    // [`Run::bring_up_containers`](crate::Run).
                    let engine = mirage_container::Engine::with_provider(&targets.provider);
                    // A `--workdir` in a containerised session names a
                    // path in the *image's* filesystem, so the host-side
                    // check `mirage run` and `mirage exec` do before they
                    // ever reach here cannot answer it — and skips it for
                    // exactly this reason. Asked of the container
                    // instead, before a single process is started, so a
                    // mistyped directory is `--workdir /wrok: there is no
                    // such directory inside container …` rather than an
                    // OCI runtime error about `chdir to cwd` and a
                    // container id the user has never seen. A container
                    // that cannot answer is believed; see
                    // [`Engine::check_workdir`](mirage_container::Engine::check_workdir).
                    if let Some(workdir) = args.workdir.as_deref()
                        && asked.insert(workdir)
                    {
                        engine
                            .check_workdir(&container, workdir)
                            .map_err(|e| MirageError::other(e.to_string()))?;
                    }
                    let mut env = env;
                    // Inside the container the host's runtime directory is
                    // a path that does not exist; what is mounted there is
                    // the session scratch, under the same name the
                    // container itself was given (see `plan_container`).
                    // Re-asserted here rather than left to the container's
                    // own environment for the same reason `LD_LIBRARY_PATH`
                    // is: `provider exec -e` replaces the container's value
                    // for that process, so a variable mirage sets on both
                    // has to be set consistently on both.
                    //
                    // The host-side attribution a containerised session
                    // needs is on the provider client below and on the
                    // container's own `mirage.runtime` label, neither of
                    // which is reachable from in here.
                    env.insert(ENV_RUNTIME.to_string(), CONTAINER_RUNTIME_DIR.to_string());
                    let env_pairs: Vec<(String, String)> = env.into_iter().collect();
                    let pid_file = targets
                        .scratch
                        .join("exec")
                        .join(exec_id.as_str())
                        .join(format!("{global}.pid"));
                    // Start from no file at all, so `ContainerProc::pid`
                    // cannot read one left by an earlier exec that
                    // happened to be given this id. It has no freshness
                    // check — it cannot have one, the file is a bare
                    // number — so a stale read is indistinguishable from a
                    // fresh one, and signalling would then deliver
                    // SIGTERM/SIGKILL to whatever process now holds that
                    // pid inside the container while the real workload
                    // ran on untouched.
                    let _ = std::fs::remove_file(&pid_file);
                    let (command, rest) =
                        pid_recording_command(&args.command, &args.args, exec_id, global);
                    // No `--workdir`, no `-w`, and deliberately no
                    // default: see "Where a containerised workload
                    // starts" on [`build_specs`].
                    let argv = engine.exec_command_line(
                        &container,
                        args.workdir.as_deref(),
                        &env_pairs,
                        &command,
                        &rest,
                        tty,
                    );
                    let (command, rest) = argv
                        .split_first()
                        .map(|(c, r)| (c.clone(), r.to_vec()))
                        .unwrap_or_else(|| (targets.provider.clone(), Vec::new()));
                    SpawnSpec {
                        node: global,
                        command,
                        args: rest,
                        // The client's own environment, not the
                        // workload's — what the workload sees was passed
                        // with `-e` above. The ownership marker is set on
                        // it anyway so that a `provider exec` client
                        // stranded by a `SIGKILL`ed run is reclaimable by
                        // the same scan as a host workload; without it
                        // the client would linger until its container was
                        // removed out from under it. Both halves of the
                        // marker, because the scan needs both: a client
                        // naming only its session belongs, as far as any
                        // other mirage can tell, to no runtime directory
                        // at all.
                        env: BTreeMap::from([
                            (ENV_SESSION.to_string(), desc.session.as_str().to_string()),
                            (ENV_RUNTIME.to_string(), runtime.clone()),
                        ]),
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

/// Whether a containerised exec should be given a pseudo-terminal.
///
/// Two conditions, and both are load-bearing.
///
/// The exec has to be the interactive *shape*: one process, holding the
/// caller's stdin. A grid is captured, nobody's stdin is connected, and a
/// pty per rank would only give mirage's output labeller a second stream
/// to untangle.
///
/// And every one of the caller's three streams has to be a terminal, not
/// just stdin. `-t` merges stderr into stdout, because a pseudo-terminal
/// is one stream. When all three are the same terminal that merge is
/// unobservable — the bytes were going to the same place regardless. When
/// they are not, it is destructive: `mirage run --image X -- job > out
/// 2> err` would put the diagnostics in `out`, which is precisely the
/// redirection this module's predecessor gave up its pty to preserve.
///
/// So a redirected interactive-shaped exec keeps pipes and loses
/// `isatty`, which is the same bargain it gets on the host path, where a
/// redirected stdout is not a terminal either.
#[must_use]
fn wants_tty(stdio: StdioMode, all_streams_are_terminals: bool) -> bool {
    stdio.owns_terminal() && all_streams_are_terminals
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
    runtime: &str,
) -> BTreeMap<String, String> {
    let mut env = desc.env.clone();
    env.extend(args.env.clone());
    env.extend(proc_env(desc, global, node, local, world_size, runtime));

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
    format!(
        "{CONTAINER_RUNTIME_DIR}/exec/{}/{global}.pid",
        exec.as_str()
    )
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
///
/// The session's own fields are read from `desc` rather than passed one
/// by one: they always come from there, and unpacking them at the call
/// site only made this the widest signature in the crate.
fn proc_env(
    desc: &SessionDescription,
    global: u32,
    node: u32,
    local: u32,
    world_size: u32,
    runtime: &str,
) -> Vec<(String, String)> {
    // The rendezvous address — and the *same string* on every rank of
    // the job, which is the whole of the requirement.
    //
    // Rank 0 used to be handed `localhost` instead, on the reasoning
    // that the head reaches its own rendezvous over loopback. That is
    // true and it is beside the point: a rendezvous address is not only
    // dialled, it is compared. `torch.distributed` builds its store key
    // and NCCL its `hostname:port` identity out of the literal string,
    // so a job whose ranks spell the head two ways can form two
    // rendezvous that never meet — on a machine where every one of those
    // connections succeeded, which is the kind of failure that survives
    // a whole afternoon of looking at the network.
    //
    // `desc.head_addr` is the value every rank can be given. On a host
    // session it is `127.0.0.1`, which reaches the head from any node
    // including the head itself, since all of them are processes on this
    // machine. On a containerised one it is the head container's name,
    // which resolves on the session's own network from inside every node
    // — node 0 included, because that is its own hostname.
    //
    // Taking the literal address rather than `localhost` also settles a
    // question that used to depend on which rank was asking: `localhost`
    // resolves to `::1` ahead of `127.0.0.1` on a dual-stack host, so a
    // store bound to IPv4 was reachable by the name on some machines and
    // not on others.
    let head = &desc.head_addr;
    let head_port = desc.head_port;
    let session = &desc.session;
    vec![
        // Which session this process belongs to. Set on every workload,
        // and inherited by everything it forks, because it is the only
        // record of the association that survives the owning run being
        // `SIGKILL`ed — see [`mirage_core::reclaim`].
        (ENV_SESSION.to_string(), session.as_str().to_string()),
        // And which mirage that session belongs to. The session name is
        // meaningful only to the runtime directory that issued it, so
        // reclamation needs both to tell a crashed run's leftovers from a
        // healthy session of a mirage running elsewhere on the machine.
        //
        // Resolved, not inherited: this overrides whatever the caller
        // exported, which is also what makes it a correct *input* for a
        // nested `mirage` — a workload that runs one sees the state
        // directory of the run it is inside.
        (ENV_RUNTIME.to_string(), runtime.to_string()),
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use mirage_core::proto::ContainerTargets;
    use mirage_core::session::SessionId;

    /// A one-process-per-node session, the shape almost every test wants.
    fn desc(node_count: u32) -> SessionDescription {
        job(node_count, 1)
    }

    /// A session running `nproc` processes on each of `node_count` nodes.
    fn job(node_count: u32, nproc: u32) -> SessionDescription {
        SessionDescription {
            session: SessionId::new("s").unwrap(),
            node_count,
            nproc_per_node: nproc,
            workdir: "/work".to_string(),
            containers: None,
            env: BTreeMap::from([("EMU".to_string(), "1".to_string())]),
            ld_preload: Some("/lib/interpose.so".to_string()),
            head_addr: "mirage-s-node-0".to_string(),
            head_port: 6000,
        }
    }

    fn exec_def(nproc: u32, node: Option<u32>) -> ExecDef {
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
            node,
            clear_env: false,
        }
    }

    fn id() -> ExecId {
        ExecId::new("e-1").unwrap()
    }

    #[test]
    fn a_single_process_job_gets_the_terminal_whole() {
        // The interactive case: `mirage run -- bash` on a one-node
        // session. Its streams are the caller's own, unmediated, which is
        // the only way a shell prints a prompt and reads a keystroke.
        let specs = build_specs(&desc(1), &exec_def(1, None), &id()).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].stdio, StdioMode::Inherit { stdin: true });
    }

    #[test]
    fn a_multi_node_job_labels_every_rank_and_gives_none_of_them_stdin() {
        // Several writers on one terminal are unreadable without labels,
        // and one terminal cannot be shared between readers — so mirage
        // takes the output to prefix it and offers stdin to nobody,
        // rather than picking a rank to receive it silently.
        let specs = build_specs(&desc(3), &exec_def(1, None), &id()).unwrap();
        assert_eq!(specs.len(), 3);
        assert!(specs.iter().all(|s| s.stdio == StdioMode::Capture));
    }

    #[test]
    fn several_processes_on_one_node_are_still_multi_process() {
        // The rule is about how many processes share the terminal, not
        // how many nodes there are.
        let specs = build_specs(&job(1, 4), &exec_def(4, None), &id()).unwrap();
        assert_eq!(specs.len(), 4);
        assert!(specs.iter().all(|s| s.stdio == StdioMode::Capture));
    }

    #[test]
    fn naming_one_node_makes_the_exec_interactive() {
        // The escape hatch for a multi-node session: one node, one
        // process, so it takes the single-process branch and gets the
        // terminal. This is what `mirage exec --node 2 -- bash` is for.
        let specs = build_specs(&desc(4), &exec_def(1, Some(2)), &id()).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].stdio, StdioMode::Inherit { stdin: true });
    }

    #[test]
    fn a_named_node_keeps_the_rank_identity_of_that_node() {
        // The process has to believe it *is* node 2, or a workload
        // started this way sees a different world from its neighbours:
        // wrong rank, wrong world size, wrong rendezvous.
        let specs = build_specs(&desc(4), &exec_def(1, Some(2)), &id()).unwrap();
        let at = |k: &str| specs[0].env.get(k).cloned().unwrap_or_default();
        assert_eq!(at("MIRAGE_RANK"), "2");
        assert_eq!(at("RANK"), "2");
        assert_eq!(at("WORLD_SIZE"), "4", "the session's size, not this exec's");
        assert_eq!(at("MASTER_ADDR"), "mirage-s-node-0");
    }

    #[test]
    fn a_node_outside_the_topology_is_refused() {
        let err = build_specs(&desc(2), &exec_def(1, Some(7)), &id()).unwrap_err();
        assert!(err.to_string().contains("no node 7"), "{err}");
    }

    #[test]
    fn the_emulator_interposer_is_prepended_to_a_user_preload() {
        let mut def = exec_def(1, None);
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
        let mut def = exec_def(1, None);
        def.exec.env.insert("RANK".to_string(), "99".to_string());
        let specs = build_specs(&desc(2), &def, &id()).unwrap();
        assert_eq!(specs[1].env.get("RANK").map(String::as_str), Some("1"));
    }

    #[test]
    fn local_and_global_ranks_are_distinct_with_several_procs_per_node() {
        let specs = build_specs(&job(2, 2), &exec_def(2, None), &id()).unwrap();
        assert_eq!(specs.len(), 4);
        let at = |i: usize, k: &str| specs[i].env.get(k).cloned().unwrap_or_default();
        assert_eq!(at(3, "RANK"), "3");
        assert_eq!(at(3, "LOCAL_RANK"), "1");
        assert_eq!(at(3, "MIRAGE_RANK"), "1");
        assert_eq!(at(3, "WORLD_SIZE"), "4");
    }

    #[test]
    fn an_exec_is_numbered_in_the_jobs_grid_and_not_in_its_own() {
        // MRG3-001. `mirage exec -- cmd` into a `--num-nodes 2
        // --nproc-per-node 3` job starts one process per node — but each
        // one is a rank *of that job*, not of a two-process job of its
        // own. Numbering them 0..1 out of a world of 2 while handing them
        // the run's own `MASTER_PORT` is what makes a collective
        // mis-form rather than fail: the six ranks the run started and
        // the two the exec started rendezvous on the same port with
        // irreconcilable ideas of how many of them there are.
        let specs = build_specs(&job(2, 3), &exec_def(1, None), &id()).unwrap();
        assert_eq!(specs.len(), 2, "one process per node, as before");
        let at = |i: usize, k: &str| specs[i].env.get(k).cloned().unwrap_or_default();

        assert_eq!(at(0, "RANK"), "0");
        assert_eq!(at(0, "LOCAL_RANK"), "0");
        assert_eq!(at(0, "MIRAGE_RANK"), "0");
        // Node 1's local rank 0 is the job's rank 3, not its rank 1.
        assert_eq!(at(1, "RANK"), "3");
        assert_eq!(at(1, "LOCAL_RANK"), "0");
        assert_eq!(at(1, "MIRAGE_RANK"), "1");
        for i in 0..2 {
            assert_eq!(at(i, "WORLD_SIZE"), "6", "the job's world, not the exec's");
        }
    }

    #[test]
    fn naming_a_node_of_a_multi_process_job_gets_that_nodes_first_rank() {
        // `mirage exec --node 1 -- bash` on a three-process-per-node job.
        // One process, so it keeps the terminal — and it is rank 3, the
        // identity node 1's own first rank has.
        let specs = build_specs(&job(2, 3), &exec_def(1, Some(1)), &id()).unwrap();
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].stdio, StdioMode::Inherit { stdin: true });
        let at = |k: &str| specs[0].env.get(k).cloned().unwrap_or_default();
        assert_eq!(at("RANK"), "3");
        assert_eq!(at("LOCAL_RANK"), "0");
        assert_eq!(at("MIRAGE_RANK"), "1");
        assert_eq!(at("WORLD_SIZE"), "6");
    }

    #[test]
    fn an_exec_may_fill_as_many_of_a_nodes_slots_as_the_job_has() {
        // Asking for the job's own shape reproduces the run's grid
        // exactly, which is the equivalence `mirage exec` rests on.
        let specs = build_specs(&job(2, 3), &exec_def(3, Some(1)), &id()).unwrap();
        let ranks: Vec<String> = specs
            .iter()
            .map(|s| s.env.get("RANK").cloned().unwrap_or_default())
            .collect();
        assert_eq!(ranks, ["3", "4", "5"]);
        let locals: Vec<String> = specs
            .iter()
            .map(|s| s.env.get("LOCAL_RANK").cloned().unwrap_or_default())
            .collect();
        assert_eq!(locals, ["0", "1", "2"]);
    }

    #[test]
    fn an_exec_cannot_ask_for_more_slots_than_a_node_has() {
        // Rank `node * P + P` is the *next* node's rank 0, so allowing
        // this would put two live processes on one rank of one
        // rendezvous — the same silent mis-forming, arrived at from the
        // other direction.
        let err = build_specs(&job(2, 2), &exec_def(3, None), &id()).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("2 process(es) per node"), "{text}");
        assert!(text.contains("--nproc-per-node 3"), "{text}");
    }

    #[test]
    fn every_rank_spells_the_rendezvous_address_the_same_way() {
        // Rank 0 was given `localhost` and every other rank the head's
        // address, which are two strings for one endpoint. Frameworks
        // compare them: a `torch.distributed` store keyed on the address
        // and a NCCL identity built from it both take the ranks at their
        // word, so the disagreement forms two rendezvous rather than
        // failing to connect.
        let specs = build_specs(&job(3, 2), &exec_def(2, None), &id()).unwrap();
        let addrs: Vec<&str> = specs
            .iter()
            .filter_map(|s| s.env.get("MASTER_ADDR").map(String::as_str))
            .collect();
        assert_eq!(addrs.len(), specs.len(), "every rank must be given one");
        assert_eq!(
            addrs,
            vec!["mirage-s-node-0"; specs.len()],
            "the ranks disagree about where the rendezvous is"
        );
        // `MIRAGE_HEAD_ADDR` is the same value under mirage's own name,
        // so it splits in exactly the same way if it is computed twice.
        let heads: Vec<&str> = specs
            .iter()
            .filter_map(|s| s.env.get("MIRAGE_HEAD_ADDR").map(String::as_str))
            .collect();
        assert_eq!(heads, addrs);
    }

    #[test]
    fn a_host_session_rendezvouses_on_the_loopback_address() {
        // What the session describes, unedited — including on rank 0,
        // which is the rank that used to be handed `localhost` instead.
        // The literal address is also the one that cannot resolve to
        // `::1` on a dual-stack host while a store listens on IPv4.
        let mut d = desc(2);
        d.head_addr = "127.0.0.1".to_string();
        let specs = build_specs(&d, &exec_def(1, None), &id()).unwrap();
        for spec in &specs {
            assert_eq!(
                spec.env.get("MASTER_ADDR").map(String::as_str),
                Some("127.0.0.1"),
                "rank {} disagrees",
                spec.node
            );
        }
    }

    #[test]
    fn an_impossible_world_size_is_refused_rather_than_forked() {
        let err = build_specs(&job(2, u32::MAX), &exec_def(1, None), &id()).unwrap_err();
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
        let specs = build_specs(&d, &exec_def(1, None), &id()).unwrap();

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
    fn every_process_is_tagged_with_its_session() {
        // The tag is the only record of the association that survives the
        // owning run being `SIGKILL`ed, so `mirage cleanup` can find what
        // was stranded. See [`mirage_core::reclaim`].
        let specs = build_specs(&job(2, 2), &exec_def(2, None), &id()).unwrap();
        assert_eq!(specs.len(), 4);
        for spec in &specs {
            assert_eq!(
                spec.env.get("MIRAGE_SESSION").map(String::as_str),
                Some("s"),
                "rank {} is untagged and would be unreclaimable",
                spec.node
            );
        }
    }

    #[test]
    fn every_process_records_the_runtime_directory_that_owns_it() {
        // `owning_runtime` reads the process-wide directory
        // resolution, which another test in this binary moves under a
        // scratch root while it runs. The shared lock keeps the value
        // this test stamps and the value it asserts the same one.
        let _paths = mirage_core::paths::test_env_lock();
        // The other half of the tag. A session name means something only
        // to the mirage that issued it, so without this a `mirage
        // cleanup` running under a different `MIRAGE_RUNTIME` sees a
        // healthy workload as a crashed run's leftovers.
        let runtime = mirage_core::container::owning_runtime();
        let specs = build_specs(&job(2, 2), &exec_def(2, None), &id()).unwrap();
        for spec in &specs {
            assert_eq!(
                spec.env.get("MIRAGE_RUNTIME").map(String::as_str),
                Some(runtime.as_str()),
                "rank {} does not say which mirage owns it",
                spec.node
            );
        }
    }

    #[test]
    fn a_user_supplied_runtime_does_not_become_the_ownership_marker() {
        // `owning_runtime` reads the process-wide directory
        // resolution, which another test in this binary moves under a
        // scratch root while it runs. The shared lock keeps the value
        // this test stamps and the value it asserts the same one.
        let _paths = mirage_core::paths::test_env_lock();
        // Same rule as the rank variables: mirage's own bookkeeping wins
        // over anything the caller exported or passed with `--env`, or a
        // workload could make itself unreclaimable — or, worse, claim to
        // belong to somebody else's runtime directory.
        let mut def = exec_def(1, None);
        def.exec
            .env
            .insert("MIRAGE_RUNTIME".to_string(), "/not/mine".to_string());
        let specs = build_specs(&desc(1), &def, &id()).unwrap();
        assert_eq!(
            specs[0].env.get("MIRAGE_RUNTIME").map(String::as_str),
            Some(mirage_core::container::owning_runtime().as_str())
        );
    }

    #[test]
    fn a_containerised_provider_client_is_tagged_too() {
        // `owning_runtime` reads the process-wide directory
        // resolution, which another test in this binary moves under a
        // scratch root while it runs. The shared lock keeps the value
        // this test stamps and the value it asserts the same one.
        let _paths = mirage_core::paths::test_env_lock();
        // The workload's own tag goes into the container with `-e`, where
        // the host-side scan cannot see it. The client that proxies it is
        // a host process and would otherwise linger unreclaimable.
        let scratch = tempfile::tempdir().unwrap();
        let mut d = desc(1);
        d.containers = Some(ContainerTargets {
            provider: "podman".to_string(),
            names: vec!["mirage-s-node-0".to_string()],
            scratch: scratch.path().to_path_buf(),
        });
        let specs = build_specs(&d, &exec_def(1, None), &id()).unwrap();
        assert_eq!(
            specs[0].env.get("MIRAGE_SESSION").map(String::as_str),
            Some("s")
        );
        // And the workload inside the container is tagged through `-e`.
        assert!(
            specs[0]
                .args
                .windows(2)
                .any(|w| w[0] == "-e" && w[1] == "MIRAGE_SESSION=s"),
            "{:?}",
            specs[0].args
        );
        // The client is a host process, so its runtime marker is the host
        // directory — that is what the `/proc` scan compares against.
        assert_eq!(
            specs[0].env.get("MIRAGE_RUNTIME").map(String::as_str),
            Some(mirage_core::container::owning_runtime().as_str())
        );
        // The workload's is the in-container mount, matching what the
        // container itself was given: the host path does not exist in
        // there, and a nested mirage would resolve it to nothing.
        assert!(
            specs[0]
                .args
                .windows(2)
                .any(|w| w[0] == "-e" && w[1] == format!("MIRAGE_RUNTIME={CONTAINER_RUNTIME_DIR}")),
            "{:?}",
            specs[0].args
        );
    }

    #[test]
    fn only_a_fully_interactive_exec_asks_for_a_container_terminal() {
        // A grid has nobody's stdin connected, so a pty per rank would
        // buy nothing and give the output labeller a merged stream.
        assert!(!wants_tty(StdioMode::Capture, true));
        // A rank that inherits the streams but not stdin is not the
        // interactive one.
        assert!(!wants_tty(StdioMode::Inherit { stdin: false }, true));
        // The interactive shape, on a real terminal.
        assert!(wants_tty(StdioMode::Inherit { stdin: true }, true));
        // The interactive shape with something redirected: `-t` merges
        // stderr into stdout, so `-- job > out 2> err` would lose the
        // distinction. Pipes and no `isatty` is the same bargain the host
        // path makes for a redirected stream.
        assert!(!wants_tty(StdioMode::Inherit { stdin: true }, false));
    }

    #[test]
    fn a_workload_inherits_the_callers_environment_by_default() {
        // Mirage's parent is the terminal the user typed in, so what they
        // exported there — a token, a PYTHONPATH, a proxy — they meant
        // for the workload.
        let specs = build_specs(&desc(1), &exec_def(1, None), &id()).unwrap();
        assert!(specs[0].inherit_env);
    }

    #[test]
    fn clear_env_asks_for_an_almost_empty_environment() {
        let mut def = exec_def(1, None);
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
        let mut def = exec_def(1, None);
        def.clear_env = true;
        let specs = build_specs(&d, &def, &id()).unwrap();
        assert!(specs[0].inherit_env);
    }

    /// A provider whose `exec` simply runs what it is handed, so the
    /// workdir probe is answered by this machine's filesystem standing
    /// in for the container's. The same stand-in `mirage_container`'s
    /// own tests use, because there is no other way to hold a container
    /// filesystem still in a unit test.
    fn executing_provider(dir: &std::path::Path) -> String {
        use std::os::unix::fs::PermissionsExt as _;
        let provider = dir.join("exec-provider.sh");
        std::fs::write(
            &provider,
            "#!/bin/sh\n\
             case \"$1\" in\n\
             exec)\n\
             shift\n\
             while [ $# -gt 0 ]; do\n\
             case \"$1\" in -i|-t) shift ;; -w|-e) shift 2 ;; *) break ;; esac\n\
             done\n\
             shift\n\
             exec \"$@\" ;;\n\
             *) exit 0 ;;\n\
             esac\n",
        )
        .unwrap();
        std::fs::set_permissions(&provider, std::fs::Permissions::from_mode(0o755)).unwrap();
        provider.to_string_lossy().to_string()
    }

    /// A one-node containerised session driven by `provider`, with its
    /// scratch under `scratch`.
    fn containerised(provider: String, scratch: &std::path::Path) -> SessionDescription {
        let mut d = desc(1);
        d.containers = Some(ContainerTargets {
            provider,
            names: vec!["mirage-s-node-0".to_string()],
            scratch: scratch.to_path_buf(),
        });
        d
    }

    /// `exec_def(1, None)` asking to start in `workdir`.
    fn exec_def_in(workdir: &str) -> ExecDef {
        let mut def = exec_def(1, None);
        def.exec.workdir = Some(workdir.to_string());
        def
    }

    #[test]
    fn a_containerised_workdir_the_image_lacks_is_refused_by_name() {
        // The host-side check both callers run skips a containerised
        // session, correctly: the path is in the image's filesystem, not
        // this one. Nothing then asked the image, so the user's answer
        // came from the provider — `OCI runtime exec failed: … chdir to
        // cwd ("/nope/nope") … no such file or directory` plus a
        // container id they have never seen, which names neither the
        // flag they passed nor the reason the path being on the host is
        // beside the point.
        let dir = tempfile::tempdir().unwrap();
        let d = containerised(executing_provider(dir.path()), dir.path());
        let err = build_specs(&d, &exec_def_in("/nope/nope"), &id()).unwrap_err();

        let message = err.to_string();
        assert!(message.contains("--workdir /nope/nope"), "{message}");
        assert!(message.contains("mirage-s-node-0"), "{message}");
        // The part nobody guesses, and the reason this is not simply the
        // host check run again: which filesystem was asked.
        assert!(
            message.contains("has to exist in the *container*"),
            "{message}"
        );
    }

    #[test]
    fn a_containerised_workdir_the_image_has_is_passed_through() {
        // The check must not invent a failure for a directory that is
        // there — the probe stands in this machine's filesystem for the
        // container's, so a real directory is a real answer.
        let dir = tempfile::tempdir().unwrap();
        let d = containerised(executing_provider(dir.path()), dir.path());
        let here = dir.path().to_string_lossy().to_string();
        let specs = build_specs(&d, &exec_def_in(&here), &id()).unwrap();
        assert!(
            specs[0]
                .args
                .windows(2)
                .any(|w| w[0] == "-w" && w[1] == here),
            "{:?}",
            specs[0].args
        );
    }

    #[test]
    fn a_containerised_exec_with_no_workdir_asks_the_container_nothing() {
        // The common case, and the one that must stay a pure function of
        // its inputs: no `--workdir` means no `-w`, no probe, and the
        // image's own `WORKDIR` decides where the workload starts. See
        // "Where a containerised workload starts" on `build_specs` for
        // why the host's directory is not carried across instead.
        let dir = tempfile::tempdir().unwrap();
        // A provider that fails whatever it is asked: reaching it at all
        // would be the bug.
        let d = containerised(
            dir.path()
                .join("no-such-engine")
                .to_string_lossy()
                .to_string(),
            dir.path(),
        );
        let specs = build_specs(&d, &exec_def(1, None), &id()).unwrap();
        assert!(
            !specs[0].args.iter().any(|a| a == "-w"),
            "{:?}",
            specs[0].args
        );
    }

    #[test]
    fn a_command_with_shell_metacharacters_is_not_parsed_by_the_wrapper() {
        let (command, argv) =
            pid_recording_command("/bin/echo", &["a b; rm -rf /".to_string()], &id(), 0);
        assert_eq!(command, "/bin/sh");
        // The dangerous argument is passed positionally, never spliced
        // into the script text.
        assert_eq!(argv.last().map(String::as_str), Some("a b; rm -rf /"));
        assert!(!argv[1].contains("rm -rf"), "{:?}", argv[1]);
    }
}
