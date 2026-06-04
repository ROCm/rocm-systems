use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileMount {
    /// the path to the file on the host
    pub host_path: String,

    /// the path to mount the file in the session
    pub container_path: String,

    /// whether the file should be mounted read-only
    pub read_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortMapping {
    /// the port on the host
    /// if set to 0, the daemon will choose a random available port
    pub host_port: u16,

    /// the port in the session
    pub container_port: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkConfigDef {
    /// no network
    None,

    /// use the host network stack
    Host,

    Custom {
        name: Option<String>,
        ports: Vec<PortMapping>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerizedDef {
    /// like docker or podman
    pub provider: String,

    /// the image to use for this session
    pub image: String,

    /// extra files to mount into the session
    pub files: Vec<FileMount>,

    /// network     
    pub network: NetworkConfigDef,
}
