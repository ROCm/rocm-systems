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
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| MirageError::Json {
        path: path.to_path_buf(),
        source: e,
    })?;
    write_bytes(path, &bytes)
}

/// Atomically write raw bytes.
///
/// Also the body of [`write_json`], which differs only in producing the
/// bytes: the scratch-file dance and its failure reporting are subtle
/// enough that two copies of them drifted apart, and there is nothing
/// about JSON that wants a second one.
pub fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| MirageError::Io {
            path: parent.to_path_buf(),
            source: e,
        })?;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let landed = write_then_rename(&tmp, path, bytes);
    if landed.is_err() {
        // A failed write has no reason to leave a `.tmp.<pid>` behind in a
        // directory the user reads and edits by hand, and the file is of
        // no use to anyone: what it holds is a document that did not land.
        let _ = fs::remove_file(&tmp);
    }
    landed.map_err(|e| write_failed(path, &e))
}

/// Fill the scratch file and move it onto `dest`.
///
/// Reports the raw [`std::io::Error`]: which of the three steps failed is
/// not something the caller distinguishes, and [`write_failed`] gives all
/// three the same account, because all three mean the same thing to the
/// user — the document is not on disk.
fn write_then_rename(tmp: &Path, dest: &Path, bytes: &[u8]) -> std::io::Result<()> {
    {
        let mut f = fs::File::create(tmp)?;
        f.write_all(bytes)?;
        f.sync_all().ok();
    }
    fs::rename(tmp, dest)
}

/// Report a write that did not land, by the path the caller asked for.
///
/// The bytes go through `<path>.tmp.<pid>` so a concurrent reader never
/// sees half a document, but that name is mirage's business. Naming it in
/// the failure told the user about a file that does not exist by the time
/// they go looking for it, and gave them no way to connect it to the
/// profile they were saving — while the thing they can actually act on,
/// the destination and the directory holding it, went unmentioned. Every
/// one of these failures is a property of that directory: it is not
/// writable, or the filesystem behind it is full or read-only.
fn write_failed(dest: &Path, source: &std::io::Error) -> MirageError {
    match dest.parent().filter(|p| !p.as_os_str().is_empty()) {
        Some(dir) => MirageError::other(format!(
            "could not write {}: {source}. Check that {} is writable and that the \
             filesystem holding it is neither full nor mounted read-only.",
            dest.display(),
            dir.display()
        )),
        None => MirageError::other(format!("could not write {}: {source}", dest.display())),
    }
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
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

    #[test]
    fn a_failed_write_names_the_document_and_not_the_scratch_file() {
        // Saving a profile into a directory the user cannot write used to
        // fail with `io error on /…/cdna4.tmp.4711` — a path that had
        // never existed as far as they were concerned, does not exist by
        // the time they look, and names neither the document they were
        // saving nor anything they could fix.
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().unwrap();
        let readonly = dir.path().join("readonly");
        fs::create_dir(&readonly).unwrap();
        fs::set_permissions(&readonly, fs::Permissions::from_mode(0o500)).unwrap();

        let dest = readonly.join("cdna4.json");
        let err = write_json(
            &dest,
            &Sample {
                a: 1,
                b: "x".into(),
            },
        )
        .unwrap_err()
        .full_message();
        assert!(err.contains("cdna4.json"), "{err}");
        assert!(!err.contains(".tmp."), "the scratch name is ours: {err}");
        assert!(err.contains(&readonly.display().to_string()), "{err}");
        assert!(err.contains("writable"), "what to check: {err}");
        assert!(err.contains("Permission denied"), "the reason: {err}");

        // And a write that did not land leaves nothing behind in a
        // directory the user reads and edits by hand.
        assert_eq!(fs::read_dir(&readonly).unwrap().count(), 0);

        fs::set_permissions(&readonly, fs::Permissions::from_mode(0o700)).unwrap();
    }
}
