// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#![allow(dead_code)]
mod amdsmi_wrapper;

#[macro_use]
mod utils;
mod amdsmi;

pub use utils::*;
pub use amdsmi::*;

#[cfg(test)]
mod tests {
    use crate::{AmdsmiLinkStatusT, AmdsmiLinkTopologyT};

    #[test]
    fn link_status_is_reexported() {
        use AmdsmiLinkStatusT::*;

        let statuses = [
            (AmdsmiLinkStatusEnabled, 0),
            (AmdsmiLinkStatusDisabled, 1),
            (AmdsmiLinkStatusInactive, 2),
            (AmdsmiLinkStatusError, 3),
        ];
        for (status, value) in statuses {
            assert_eq!(status as u32, value);
        }
        let _: fn(&AmdsmiLinkTopologyT) -> AmdsmiLinkStatusT = |topology| topology.link_status;
    }
}
