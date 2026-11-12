// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Exercises zero-sized storage owners through the production library build.

use rocddi::memory::storage::{AllocationError, Owned, Shared};

#[test]
fn zero_sized_owners_use_the_production_storage_layout() -> Result<(), AllocationError> {
    drop(Owned::new(())?);
    let () = Owned::new(())?.into_inner();

    let shared = Shared::new(())?;
    let clone = shared.clone();
    drop(shared);
    drop(clone);
    Ok(())
}
