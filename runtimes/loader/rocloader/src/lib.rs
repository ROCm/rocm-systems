// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD HSA code-object loading and executable ownership.
//!
//! A private, safe ELF module supplies checked byte operations. This crate
//! interprets AMD HSA code objects, builds device images through `rocddi`, and owns
//! executable state. Frontends supply device selection, opaque handle IDs,
//! source URIs, and their public ABI translations.

mod code_object;
mod elf;
mod executable;
mod loaded_image;
mod target;

#[cfg(test)]
mod test_support;

pub use code_object::{
    CodeObject, CodeObjectContract, FloatRoundingMode, Profile, Symbol, SymbolKind, SymbolLinkage,
    SymbolTableOrder, Target, VariableAllocation, VariableSegment, supports_target,
};
pub use executable::{Definition, Executable, LoadedObject, Scope};
pub use loaded_image::{
    ImagePublication, LoadedImage, LoadedSegment, load_image, load_image_at_address,
    load_image_with_peers,
};

/// Failure while parsing, loading, or changing an executable.
#[derive(Debug)]
pub enum Error {
    /// The code object or its load image is malformed or unsupported.
    InvalidObject,
    /// An external symbol needed for loading has no definition.
    UndefinedSymbol,
    /// The code object's machine identifier is unknown.
    InvalidTarget,
    /// The code object cannot execute on the selected device.
    IncompatibleTarget,
    /// An allocation extent or address cannot be represented.
    OutOfResources,
    /// A native device operation failed.
    Device(rocddi::Error),
    /// An address lies within a loaded object but its requested range does not.
    InvalidRange,
    /// A frozen executable cannot be changed.
    Frozen,
    /// The program object has already been loaded.
    ProgramAlreadyLoaded,
    /// An executable already contains the same symbol in the same scope.
    AlreadyDefined,
    /// A publication is in progress or its native retirement is uncertain.
    Busy,
}

impl From<elf::Error> for Error {
    fn from(error: elf::Error) -> Self {
        match error {
            elf::Error::InvalidObject => Self::InvalidObject,
            elf::Error::UndefinedSymbol => Self::UndefinedSymbol,
        }
    }
}
