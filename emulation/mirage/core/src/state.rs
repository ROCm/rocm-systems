//! On-disk readers/writers for mirage state.
//!
//! Writes are performed atomically (write-then-rename) so that readers
//! never observe a truncated file.

use std::fs;
use std::io::Write;
use std::path::Path;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::error::{MirageError, Result};

/// Read a JSON document from a file.
pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).map_err(|e| MirageError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    serde_json::from_slice(&bytes).map_err(|e| MirageError::Json {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Read a JSON document if the file exists; return `Ok(None)` if not.
pub fn read_json_opt<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match fs::read(path) {
        Ok(bytes) => Ok(Some(serde_json::from_slice(&bytes).map_err(|e| {
            MirageError::Json {
                path: path.to_path_buf(),
                source: e,
            }
        })?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(MirageError::Io {
            path: path.to_path_buf(),
            source: e,
        }),
    }
}

/// Atomically write a JSON document.
///
/// The serialized bytes are written to `<path>.tmp.<pid>` and then
/// renamed onto `path`. Parent directories are created as needed.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| MirageError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| MirageError::Json {
        path: path.to_path_buf(),
        source: e,
    })?;
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp).map_err(|e| MirageError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.write_all(&bytes).map_err(|e| MirageError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.sync_all().ok();
    }
    fs::rename(&tmp, path).map_err(|e| MirageError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Atomically write raw bytes.
pub fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| MirageError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    {
        let mut f = fs::File::create(&tmp).map_err(|e| MirageError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.write_all(bytes).map_err(|e| MirageError::Io {
            path: tmp.clone(),
            source: e,
        })?;
        f.sync_all().ok();
    }
    fs::rename(&tmp, path).map_err(|e| MirageError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

/// Read a small "value" file as a trimmed string (or return `None`).
pub fn read_small_str(path: &Path) -> Result<Option<String>> {
    match fs::read_to_string(path) {
        Ok(s) => Ok(Some(s.trim().to_string())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(MirageError::Io {
            path: path.to_path_buf(),
            source: e,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Sample {
        a: u32,
        b: String,
    }

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x/y.json");
        let s = Sample {
            a: 7,
            b: "hi".to_string(),
        };
        write_json(&p, &s).unwrap();
        let r: Sample = read_json(&p).unwrap();
        assert_eq!(r, s);
    }

    #[test]
    fn read_missing() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("nope.json");
        let r: Option<Sample> = read_json_opt(&p).unwrap();
        assert!(r.is_none());
    }
}
