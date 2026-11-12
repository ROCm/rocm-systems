// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD HSA code-object loading and executable ownership.
//!
//! The ELF parser remains independent in `rocelf`. This crate interprets AMD
//! HSA code objects, builds device-resident images through `rocddi`, and owns
//! executable state. Frontends supply device selection, opaque handle IDs,
//! source URIs, and their public ABI translations.

mod code_object;
mod executable;
mod image;

pub use code_object::{
    CodeObject, Symbol, SymbolKind, SymbolLinkage, SymbolTableOrder, Target, VariableAllocation,
    VariableSegment,
};
pub use executable::{Definition, Executable, LoadedObject, Scope};
pub use image::{LoadedImage, LoadedSegment, load_image, loaded_segments};

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
}

impl From<rocelf::Error> for Error {
    fn from(error: rocelf::Error) -> Self {
        match error {
            rocelf::Error::InvalidObject => Self::InvalidObject,
            rocelf::Error::UndefinedSymbol => Self::UndefinedSymbol,
        }
    }
}
