//! Installs the mirage web UI as a systemd service.
//!
//! `mirage webui install` writes a `mirage-webui.service` unit that runs
//! `mirage webui --addr <addr>` and (optionally) enables/starts it.
//! By default it installs a per-user unit under
//! `~/.config/systemd/user`; `--system` installs a system-wide unit
//! under `/etc/systemd/system`.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, bail};
use clap::Args;

/// The systemd unit file name installed by `mirage webui install`.
const UNIT_NAME: &str = "mirage-webui.service";

/// Command-line flags for `mirage webui install`.
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

/// Install (or print) the systemd unit for the web UI.
pub fn install(addr: SocketAddr, args: &InstallArgs) -> anyhow::Result<()> {
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
fn render_unit(exe: &std::path::Path, addr: SocketAddr, system: bool) -> String {
    let wanted_by = if system {
        "multi-user.target"
    } else {
        "default.target"
    };
    format!(
        "[Unit]\n\
         Description=Mirage web UI\n\
         After=network-online.target\n\
         Wants=network-online.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart={exe} webui --addr {addr}\n\
         Restart=on-failure\n\
         RestartSec=2\n\
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
