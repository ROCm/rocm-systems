// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use crate::ffi::*;

#[test]
/// Verifies that native busy reports resource contention across frontends.
fn native_busy_reports_resource_contention_across_frontends() {
    let busy = || rocddi::Error::Operation {
        kind: rocddi::ErrorKind::Busy,
        detail: "injected live native dependency",
    };
    assert_eq!(map_error(busy()), RESOURCE_BUSY);
    assert_ne!(map_error(busy()), INVALID_QUEUE);
}

#[test]
/// Unsupported native services retain an actionable public status.
fn unsupported_native_services_report_not_supported() {
    assert_eq!(
        map_error(rocddi::Error::Operation {
            kind: rocddi::ErrorKind::Unsupported,
            detail: "unqualified transport",
        }),
        NOT_SUPPORTED
    );
}
