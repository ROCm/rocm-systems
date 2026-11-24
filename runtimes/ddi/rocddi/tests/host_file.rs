// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native host file contract, runnable without a device provider.
#![allow(clippy::unwrap_used)]

use rocddi::host::file;
use std::io::{Read, Seek, SeekFrom, Write};

#[test]
fn positioned_io_rejects_offsets_outside_the_signed_file_range() {
    let path = std::env::temp_dir().join(format!("rocddi-host-file-offset-{}", std::process::id()));
    let mut caller = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    caller.write_all(b"unchanged").unwrap();
    caller.seek(SeekFrom::Start(2)).unwrap();

    for offset in [i64::MAX as u64 + 1, u64::MAX] {
        assert_eq!(
            file::write_at(&caller, b"X", offset).unwrap_err().kind(),
            std::io::ErrorKind::InvalidInput
        );
        let mut bytes = [0x5a];
        assert_eq!(
            file::read_at(&caller, &mut bytes, offset)
                .unwrap_err()
                .kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert_eq!(bytes, [0x5a]);
        assert_eq!(caller.stream_position().unwrap(), 2);
        assert_eq!(caller.metadata().unwrap().len(), 9);
        assert_eq!(std::fs::read(&path).unwrap(), b"unchanged");
    }

    drop(caller);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn positioned_io_keeps_the_original_cursor_and_native_owner() {
    let path = std::env::temp_dir().join(format!("rocddi-host-file-{}", std::process::id()));
    let mut caller = std::fs::OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    caller.write_all(b"abcdef").unwrap();
    caller.seek(SeekFrom::Start(2)).unwrap();
    #[cfg(unix)]
    let retained = {
        use std::os::fd::AsRawFd;
        rocddi::host::linux::duplicate_file(caller.as_raw_fd()).unwrap()
    };
    #[cfg(windows)]
    let retained = {
        use std::os::windows::io::AsRawHandle;
        rocddi::host::windows::duplicate_file(caller.as_raw_handle()).unwrap()
    };
    let mut bytes = [0; 3];
    file::read_exact_at(&retained, &mut bytes, 1).unwrap();
    assert_eq!(&bytes, b"bcd");
    assert_eq!(file::write_at(&retained, b"Z", 5).unwrap(), 1);
    assert_eq!(caller.stream_position().unwrap(), 2);
    assert_eq!(
        file::read_exact_at(&retained, &mut bytes, 5)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::UnexpectedEof
    );
    assert_eq!(
        file::read_exact_at(&retained, &mut bytes, u64::MAX)
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
    drop(retained);
    caller.read_exact(&mut bytes).unwrap();
    assert_eq!(&bytes, b"cde");
    drop(caller);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn positioned_io_requires_the_original_handle_access() {
    let path = std::env::temp_dir().join(format!("rocddi-host-file-access-{}", std::process::id()));
    std::fs::write(&path, b"unchanged").unwrap();
    let reader = std::fs::File::open(&path).unwrap();
    assert!(file::write_at(&reader, b"X", 0).is_err());
    let writer = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    assert!(file::read_at(&writer, &mut [0; 1], 0).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"unchanged");
    drop(reader);
    drop(writer);
    std::fs::remove_file(path).unwrap();
}
