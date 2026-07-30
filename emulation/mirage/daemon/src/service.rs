//! Installs the mirage supervisor daemon as a systemd service.
//!
//! `mirage daemon install` writes a `mirage.service` unit that runs
//! `mirage daemon` and (optionally) enables and starts it. By default it
//! installs a per-user unit under `~/.config/systemd/user`; `--system`
//! installs a system-wide unit under `/etc/systemd/system`.
//!
//! The unit disables the idle timeout: a service is expected to stay
//! running, and exiting after ten idle minutes would leave systemd
//! restarting it in a loop. The CLI's auto-start path is where the idle
//! timeout belongs.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, bail};
use clap::Args;

/// The systemd unit file name installed by `mirage daemon install`.
const UNIT_NAME: &str = "mirage.service";

/// Command-line flags for `mirage daemon install`.
#[derive(Args, Debug, Clone)]
pub struct InstallArgs {
    /// Install a system-wide unit under `/etc/systemd/system` instead
    /// of a per-user unit under `~/.config/systemd/user`.
    #[arg(long)]
    pub system: bool,

    /// Reload systemd and `enable --now` the service after writing it.
    #[arg(long)]
    pub enable: bool,

    /// Print the generated unit to stdout instead of writing any files.
    #[arg(long)]
    pub print: bool,
}

/// Install (or print) the systemd unit for the daemon.
///
/// `addr`, when set, additionally serves the HTTP dashboard.
///
/// # Errors
///
/// Returns an error if the executable path cannot be determined, the unit
/// cannot be written, or `systemctl` fails.
pub(crate) fn install(addr: Option<SocketAddr>, args: &InstallArgs) -> anyhow::Result<()> {
    let exe = std::env::current_exe()
        .context("could not determine the path to the running mirage executable")?;
    let unit = render_unit(&exe, addr, args.system);

    if args.print {
        print!("{unit}");
        return Ok(());
    }

    let path = unit_path(args.system)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating unit directory {}", parent.display()))?;
    }
    std::fs::write(&path, &unit)
        .with_context(|| format!("writing systemd unit {}", path.display()))?;
    println!("installed {}", path.display());

    if args.enable {
        enable_service(args.system)?;
        println!("enabled and started {UNIT_NAME}");
    } else {
        print_manual_steps(args.system);
    }
    Ok(())
}

/// Render the unit file contents for the given executable and address.
fn render_unit(exe: &std::path::Path, addr: Option<SocketAddr>, system: bool) -> String {
    let wanted_by = if system {
        "multi-user.target"
    } else {
        "default.target"
    };
    let http = addr.map(|a| format!(" --addr {a}")).unwrap_or_default();
    format!(
        "[Unit]\n\
         Description=Mirage supervisor daemon\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe} daemon --idle-timeout 0{http}\n\
         Restart=on-failure\n\
         RestartSec=2\n\
         # Give the daemon time to tear every session down. It kills each\n\
         # workload's process group and waits for it, so a shorter stop\n\
         # timeout would have systemd SIGKILL the daemon mid-cleanup and\n\
         # orphan exactly the processes it was in the middle of reaping.\n\
         #\n\
         # KillMode=mixed is what makes that ordering the daemon's to\n\
         # keep: the default, control-group, SIGTERMs every process in\n\
         # the cgroup at once, so the workloads and the emulator daemon\n\
         # die alongside the supervisor instead of in the order it tears\n\
         # them down. With mixed, only the supervisor is signalled and\n\
         # systemd falls back to SIGKILLing the group after the timeout.\n\
         KillSignal=SIGTERM\n\
         KillMode=mixed\n\
         TimeoutStopSec=60\n\
         \n\
         [Install]\n\
         WantedBy={wanted_by}\n",
        exe = exe.display(),
    )
}

/// Resolve the path the unit should be written to.
fn unit_path(system: bool) -> anyhow::Result<PathBuf> {
    if system {
        return Ok(PathBuf::from("/etc/systemd/system").join(UNIT_NAME));
    }
    let base = match std::env::var_os("XDG_CONFIG_HOME") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => {
            let home = std::env::var_os("HOME")
                .filter(|h| !h.is_empty())
                .context("HOME is not set; cannot locate ~/.config for the user unit")?;
            PathBuf::from(home).join(".config")
        }
    };
    Ok(base.join("systemd").join("user").join(UNIT_NAME))
}

/// Run `systemctl daemon-reload` and `enable --now`.
fn enable_service(system: bool) -> anyhow::Result<()> {
    run_systemctl(system, &["daemon-reload"])?;
    run_systemctl(system, &["enable", "--now", UNIT_NAME])?;
    Ok(())
}

/// Invoke `systemctl` (with `--user` for user units), surfacing failures.
fn run_systemctl(system: bool, args: &[&str]) -> anyhow::Result<()> {
    let mut cmd = Command::new("systemctl");
    if !system {
        cmd.arg("--user");
    }
    cmd.args(args);
    let status = cmd
        .status()
        .context("failed to run `systemctl`; is systemd available on this host?")?;
    if !status.success() {
        let scope = if system { "" } else { "--user " };
        bail!("`systemctl {scope}{}` failed", args.join(" "));
    }
    Ok(())
}

/// Print the manual `systemctl` steps when `--enable` was not given.
fn print_manual_steps(system: bool) {
    let user = if system { "" } else { "--user " };
    println!("to start it now, run:");
    println!("  systemctl {user}daemon-reload");
    println!("  systemctl {user}enable --now {UNIT_NAME}");
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn the_unit_disables_the_idle_timeout() {
        // A service that exits when idle would have systemd restart it
        // forever.
        let unit = render_unit(std::path::Path::new("/usr/bin/mirage"), None, true);
        assert!(unit.contains("--idle-timeout 0"), "{unit}");
    }

    #[test]
    fn the_unit_allows_enough_time_to_tear_sessions_down() {
        // systemd's default TimeoutStopSec would SIGKILL the daemon
        // partway through reaping workload process groups, orphaning
        // them — the exact failure this whole design removes.
        let unit = render_unit(std::path::Path::new("/usr/bin/mirage"), None, false);
        assert!(unit.contains("KillSignal=SIGTERM"), "{unit}");
        assert!(unit.contains("TimeoutStopSec=60"), "{unit}");
        // Without this, systemd's default KillMode=control-group
        // SIGTERMs the workloads at the same instant as the supervisor,
        // pre-empting the ordered teardown the timeout above exists to
        // protect.
        assert!(unit.contains("KillMode=mixed"), "{unit}");
    }

    #[test]
    fn the_http_address_is_only_passed_when_requested() {
        let without = render_unit(std::path::Path::new("/usr/bin/mirage"), None, false);
        assert!(!without.contains("--addr"), "{without}");

        let with = render_unit(
            std::path::Path::new("/usr/bin/mirage"),
            Some("127.0.0.1:5174".parse().unwrap()),
            false,
        );
        assert!(with.contains("--addr 127.0.0.1:5174"), "{with}");
    }

    #[test]
    fn a_user_unit_targets_the_user_session() {
        let user = render_unit(std::path::Path::new("/usr/bin/mirage"), None, false);
        assert!(user.contains("WantedBy=default.target"), "{user}");
        let system = render_unit(std::path::Path::new("/usr/bin/mirage"), None, true);
        assert!(system.contains("WantedBy=multi-user.target"), "{system}");
    }
}
