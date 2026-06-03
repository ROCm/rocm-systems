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
pub struct ContainerizedDef {
    /// like docker or podman
    pub provider: String,

    /// the image to use for this session
    pub image: String,

    /// extra files to mount into the session
    pub mounts: Vec<FileMount>,
}
