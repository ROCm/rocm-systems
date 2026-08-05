//! Profile: a reusable bundle of emulator configuration.

use serde::{Deserialize, Serialize};

use crate::emulator::EmulatorDef;

/// A single host↔container bind mount.
///
/// Mounts are applied when a node's container is created (`-v
/// HOST:CONTAINER[:ro]`). They are part of a [`ContainerizedDef`] and
/// therefore live on the profile, so a profile fully describes how its
/// sessions are containerised.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMount {
    /// Path to the file or directory on the host.
    pub host_path: String,

    /// Path the file or directory is mounted at inside the container.
    pub container_path: String,

    /// Whether the mount is read-only.
    #[serde(default)]
    pub read_only: bool,
}

impl FileMount {
    /// Parse a CLI `--mount` spec.
    ///
    /// Accepted forms:
    /// * `HOST` — mount `HOST` at the same path, read-write.
    /// * `HOST:CONTAINER` — bind `HOST` to `CONTAINER`, read-write.
    /// * `HOST:CONTAINER:ro` — as above, read-only (`rw` is also accepted).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let parts: Vec<&str> = spec.split(':').collect();
        match parts.as_slice() {
            [host] if !host.is_empty() => Ok(FileMount {
                host_path: (*host).to_string(),
                container_path: (*host).to_string(),
                read_only: false,
            }),
            [host, container] if !host.is_empty() && !container.is_empty() => Ok(FileMount {
                host_path: (*host).to_string(),
                container_path: (*container).to_string(),
                read_only: false,
            }),
            [host, container, mode] if !host.is_empty() && !container.is_empty() => {
                let read_only = match *mode {
                    "ro" => true,
                    "rw" => false,
                    other => {
                        return Err(format!(
                            "invalid mount mode {other:?} in {spec:?} (expected `ro` or `rw`)"
                        ));
                    }
                };
                Ok(FileMount {
                    host_path: (*host).to_string(),
                    container_path: (*container).to_string(),
                    read_only,
                })
            }
            _ => Err(format!(
                "invalid mount spec {spec:?} (expected HOST[:CONTAINER[:ro|rw]])"
            )),
        }
    }

    /// Render the `-v` argument value (`HOST:CONTAINER[:ro]`) used by
    /// the container provider.
    pub fn to_volume_arg(&self) -> String {
        if self.read_only {
            format!("{}:{}:ro", self.host_path, self.container_path)
        } else {
            format!("{}:{}", self.host_path, self.container_path)
        }
    }
}

/// A single host→container published port (docker `-p` / podman
/// `--publish`).
///
/// Ports are applied when a node's container is created (`-p
/// HOST:CONTAINER[/PROTO]`). Like [`FileMount`]s they are part of a
/// [`ContainerizedDef`] and therefore live on the profile, so a profile
/// fully describes which container ports it exposes on the host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortMapping {
    /// Port published on the host.
    pub host_port: u16,

    /// Port the container listens on.
    pub container_port: u16,

    /// Optional protocol (`tcp` or `udp`). `None` lets the provider
    /// default (tcp).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol: Option<String>,
}

impl PortMapping {
    /// Parse a CLI `--port` spec.
    ///
    /// Accepted forms:
    /// * `PORT` — publish the same port on host and container.
    /// * `HOST:CONTAINER` — publish container port `CONTAINER` as host
    ///   port `HOST`.
    /// * either form with a `/tcp` or `/udp` suffix to pin the protocol.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let (ports, protocol) = match spec.split_once('/') {
            Some((ports, proto)) => {
                let proto = match proto {
                    "tcp" | "udp" => proto.to_string(),
                    other => {
                        return Err(format!(
                            "invalid port protocol {other:?} in {spec:?} (expected `tcp` or `udp`)"
                        ));
                    }
                };
                (ports, Some(proto))
            }
            None => (spec, None),
        };

        let parse_port = |s: &str| -> Result<u16, String> {
            s.parse::<u16>()
                .map_err(|_| format!("invalid port {s:?} in {spec:?} (expected a number 1-65535)"))
        };

        let (host_port, container_port) = match ports.split_once(':') {
            Some((host, container)) if !host.is_empty() && !container.is_empty() => {
                (parse_port(host)?, parse_port(container)?)
            }
            Some(_) => {
                return Err(format!(
                    "invalid port spec {spec:?} (expected HOST_PORT[:CONTAINER_PORT][/tcp|/udp])"
                ));
            }
            None if !ports.is_empty() => {
                let p = parse_port(ports)?;
                (p, p)
            }
            None => {
                return Err(format!(
                    "invalid port spec {spec:?} (expected HOST_PORT[:CONTAINER_PORT][/tcp|/udp])"
                ));
            }
        };

        Ok(PortMapping {
            host_port,
            container_port,
            protocol,
        })
    }

    /// Render the `-p` argument value (`HOST:CONTAINER[/PROTO]`) used by
    /// the container provider.
    pub fn to_publish_arg(&self) -> String {
        match &self.protocol {
            Some(proto) => format!("{}:{}/{proto}", self.host_port, self.container_port),
            None => format!("{}:{}", self.host_port, self.container_port),
        }
    }
}

/// An opt-in environment "hack": a best-effort workaround for image
/// incompatibilities, applied by building a derivative image from the
/// profile's base image before launching any node containers.
///
/// Hacks are explicit and additive (a profile may carry several). They
/// are deliberately scoped to containerised sessions, where mirage owns
/// the image build, and are a no-op otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hack {
    /// Update GCC's runtime libraries (`libstdc++6`/`libgcc-s1`) in the
    /// base image via the `ubuntu-toolchain-r/test` PPA. Resolves
    /// `GLIBCXX_*`/`GCC_*` "version not found" failures from binaries
    /// (e.g. the bind-mounted mirage host and emulator interposers)
    /// built against a newer toolchain than the image ships.
    UpdateGccViaPpa,
}

impl Hack {
    /// Stable slug used in derived-image tags and logs.
    pub fn slug(self) -> &'static str {
        match self {
            Hack::UpdateGccViaPpa => "update-gcc-via-ppa",
        }
    }

    /// The Dockerfile `RUN` step that realises this hack, layered on top
    /// of the base image. Each is written to be idempotent and to clean
    /// up package-manager caches so the derived image stays lean.
    pub fn dockerfile_step(self) -> &'static str {
        match self {
            // Pull a newer libstdc++/libgcc from the well-known Ubuntu
            // toolchain PPA. `add-apt-repository` is broken in some
            // containers (its Python can't import `apt_pkg`), so add the
            // PPA manually via curl + a dearmored keyring — mirroring
            // emulation/rocjitsu/scripts/rocjitsu-docker-build.sh.
            // `DEBIAN_FRONTEND=noninteractive` keeps apt from prompting.
            Hack::UpdateGccViaPpa => {
                "RUN export DEBIAN_FRONTEND=noninteractive \\\n \
                 && apt-get update \\\n \
                 && apt-get install -y --no-install-recommends curl gnupg ca-certificates \\\n \
                 && mkdir -p /root/.gnupg && chmod 700 /root/.gnupg \\\n \
                 && curl -fsSL \"https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x1E9377A2BA9EF27F\" \\\n \
                 | gpg --dearmor > /usr/share/keyrings/toolchain.gpg \\\n \
                 && echo \"deb [signed-by=/usr/share/keyrings/toolchain.gpg] http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu jammy main\" \\\n \
                 > /etc/apt/sources.list.d/toolchain.list \\\n \
                 && apt-get update \\\n \
                 && apt-get install -y --only-upgrade libstdc++6 libgcc-s1 \\\n \
                 && rm -rf /var/lib/apt/lists/*"
            }
        }
    }
}

/// Generate the Dockerfile that builds a derivative of `base` with each
/// of `hacks` applied as an additional layer, in order. Returns `None`
/// when `hacks` is empty (no derivative image is needed).
pub fn hacks_dockerfile(base: &str, hacks: &[Hack]) -> Option<String> {
    if hacks.is_empty() {
        return None;
    }
    let mut dockerfile = format!("FROM {base}\n");
    for hack in hacks {
        dockerfile.push_str(hack.dockerfile_step());
        dockerfile.push('\n');
    }
    Some(dockerfile)
}

/// Deterministic tag for the derivative image built from `base` with
/// `hacks` applied. The tag is a pure function of the base image and the
/// (order-independent) set of hacks, so repeated runs reuse a previously
/// built image instead of rebuilding it. Returns `None` when `hacks` is
/// empty.
pub fn hacks_image_tag(base: &str, hacks: &[Hack]) -> Option<String> {
    if hacks.is_empty() {
        return None;
    }
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut slugs: Vec<&str> = hacks.iter().map(|h| h.slug()).collect();
    slugs.sort_unstable();
    slugs.dedup();
    let mut hasher = DefaultHasher::new();
    base.hash(&mut hasher);
    slugs.hash(&mut hasher);
    Some(format!("mirage-hack-{:016x}:latest", hasher.finish()))
}

/// Containerisation settings for a profile.
///
/// When a profile carries a `ContainerizedDef`, every node of a session
/// using that profile runs inside its own container; the containers are
/// joined by a per-session virtual network so nodes can reach each
/// other (and the head) by name. See [`crate::container`] for the
/// runtime types and `mirage_container` for the engine that drives the
/// provider CLI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerizedDef {
    /// Provider binary to drive: `"podman"`, `"docker"`, or an absolute
    /// path. `None` auto-detects (preferring podman, then docker).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,

    /// Image reference to run each node in (e.g. `rocm/dev:latest`).
    pub image: String,

    /// Extra bind mounts applied to every node container.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mounts: Vec<FileMount>,

    /// Ports published from every node container to the host (`-p
    /// HOST:CONTAINER[/PROTO]`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<PortMapping>,

    /// Host device nodes to expose to every node container (`--device`),
    /// e.g. `/dev/kfd` and `/dev/dri` for AMD GPU access.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub devices: Vec<String>,

    /// Supplementary groups to add inside every node container
    /// (`--group-add`), e.g. `video`/`render` so the workload may open
    /// the GPU device nodes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<String>,

    /// Opt-in image [`Hack`]s applied by building a derivative image
    /// from `image` before launching node containers. Empty for the
    /// common case (run the image as-is).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hacks: Vec<Hack>,
}

/// A profile is a named, on-disk emulator preset that can be referenced
/// by sessions. Profiles live in `$XDG_CONFIG_HOME/mirage/profile/`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileDef {
    /// The profile's name (also used as the filename, `<name>.json`).
    pub name: String,

    /// Free-form description shown in `mirage profile list -l`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// The emulator to use for this profile.
    pub emulator: EmulatorDef,

    /// Optionally containerise the session: when set, every node runs
    /// inside its own container on a shared per-session network.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub containerize: Option<ContainerizedDef>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mount_parse_host_only() {
        let m = FileMount::parse("/data").unwrap();
        assert_eq!(m.host_path, "/data");
        assert_eq!(m.container_path, "/data");
        assert!(!m.read_only);
    }

    #[test]
    fn mount_parse_host_container() {
        let m = FileMount::parse("/h:/c").unwrap();
        assert_eq!(m.host_path, "/h");
        assert_eq!(m.container_path, "/c");
        assert!(!m.read_only);
        assert_eq!(m.to_volume_arg(), "/h:/c");
    }

    #[test]
    fn mount_parse_readonly() {
        let m = FileMount::parse("/h:/c:ro").unwrap();
        assert!(m.read_only);
        assert_eq!(m.to_volume_arg(), "/h:/c:ro");
    }

    #[test]
    fn mount_parse_rejects_bad_mode() {
        assert!(FileMount::parse("/h:/c:xx").is_err());
        assert!(FileMount::parse("").is_err());
    }

    #[test]
    fn port_parse_host_container() {
        let p = PortMapping::parse("8080:8000").unwrap();
        assert_eq!(p.host_port, 8080);
        assert_eq!(p.container_port, 8000);
        assert_eq!(p.protocol, None);
        assert_eq!(p.to_publish_arg(), "8080:8000");
    }

    #[test]
    fn port_parse_single() {
        let p = PortMapping::parse("8000").unwrap();
        assert_eq!(p.host_port, 8000);
        assert_eq!(p.container_port, 8000);
        assert_eq!(p.to_publish_arg(), "8000:8000");
    }

    #[test]
    fn port_parse_with_protocol() {
        let p = PortMapping::parse("53:53/udp").unwrap();
        assert_eq!(p.host_port, 53);
        assert_eq!(p.container_port, 53);
        assert_eq!(p.protocol.as_deref(), Some("udp"));
        assert_eq!(p.to_publish_arg(), "53:53/udp");
    }

    #[test]
    fn port_parse_rejects_bad_specs() {
        assert!(PortMapping::parse("").is_err());
        assert!(PortMapping::parse("notaport").is_err());
        assert!(PortMapping::parse("8080:").is_err());
        assert!(PortMapping::parse("99999:8000").is_err());
        assert!(PortMapping::parse("8080:8000/sctp").is_err());
    }
}
