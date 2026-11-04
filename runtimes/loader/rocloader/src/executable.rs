// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Executable publication, symbol scopes, and loaded-image lifetime.

use std::collections::{HashMap, HashSet};

use crate::{Error, LoadedImage, Symbol};

/// Scope in which an executable symbol is defined and queried.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Scope {
    /// One definition visible to the whole executable.
    Program,
    /// One definition visible to an activated agent identified by the frontend.
    Agent(u64),
}

/// A symbol and frontend-assigned opaque identifier staged for publication.
pub struct Definition {
    /// Interpreted symbol metadata and resolved address.
    pub symbol: Symbol,
    /// Program or agent scope.
    pub scope: Scope,
    /// Opaque symbol ID assigned by the frontend before publication.
    pub id: u64,
}

/// One owned image and the frontend's loaded-object metadata.
pub struct LoadedObject<M> {
    /// Device image retained until this loaded object is removed.
    pub image: LoadedImage,
    /// Frontend-owned metadata associated with that image.
    pub metadata: M,
}

/// Mutable AMD HSA executable with transactional object publication.
///
/// Opaque IDs and ABI metadata belong to the frontend. This owner retains
/// loaded device allocations and the scoped name-to-ID index. Its state is
/// immutable after freeze.
pub struct Executable<M> {
    /// Prevents mutation after executable publication.
    frozen: bool,
    /// Enforces the single program-scope object rule.
    program_loaded: bool,
    /// Loaded images retained for the executable's lifetime.
    loaded: Vec<LoadedObject<M>>,
    /// Symbol names indexed by visibility scope.
    names: HashMap<Scope, HashMap<String, u64>>,
    /// Resolved symbols indexed by frontend-assigned handle ID.
    symbols: HashMap<u64, Symbol>,
    /// Stable enumeration order of published symbol IDs.
    symbol_ids: Vec<u64>,
}

impl<M> Executable<M> {
    /// Creates an empty executable, optionally already frozen.
    #[must_use]
    pub fn new(frozen: bool) -> Self {
        Self {
            frozen,
            program_loaded: false,
            loaded: Vec::new(),
            names: HashMap::new(),
            symbols: HashMap::new(),
            symbol_ids: Vec::new(),
        }
    }

    /// Returns whether the executable is frozen.
    #[must_use]
    pub fn is_frozen(&self) -> bool {
        self.frozen
    }

    /// Freezes the executable and its symbol addresses.
    ///
    /// # Errors
    /// Returns `Frozen` if it was already frozen.
    pub fn freeze(&mut self) -> Result<(), Error> {
        if self.frozen {
            return Err(Error::Frozen);
        }
        self.frozen = true;
        Ok(())
    }

    /// Checks whether another object of the given scope may be loaded.
    ///
    /// # Errors
    /// Returns `Frozen` or `ProgramAlreadyLoaded` when the transition is invalid.
    pub fn can_load(&self, program: bool) -> Result<(), Error> {
        if self.frozen {
            return Err(Error::Frozen);
        }
        if program && self.program_loaded {
            return Err(Error::ProgramAlreadyLoaded);
        }
        Ok(())
    }

    /// Returns loaded objects in publication order.
    #[must_use]
    pub fn loaded(&self) -> &[LoadedObject<M>] {
        &self.loaded
    }

    /// Returns frontend-assigned symbol IDs in publication order.
    #[must_use]
    pub fn symbol_ids(&self) -> &[u64] {
        &self.symbol_ids
    }

    /// Finds the ID of a symbol in exactly one scope.
    #[must_use]
    pub fn symbol_id(&self, name: &str, scope: Scope) -> Option<u64> {
        self.names.get(&scope)?.get(name).copied()
    }

    /// Borrows a published symbol by its frontend-assigned ID.
    #[must_use]
    pub fn symbol(&self, id: u64) -> Option<&Symbol> {
        self.symbols.get(&id)
    }

    /// Resolves a program symbol first, then the selected agent symbol.
    ///
    /// A program definition takes precedence over an agent definition.
    #[must_use]
    pub fn resolve(&self, name: &str, agent: u64) -> Option<u64> {
        let id = self
            .symbol_id(name, Scope::Program)
            .or_else(|| self.symbol_id(name, Scope::Agent(agent)))?;
        self.symbol(id)
            .filter(|symbol| symbol.is_definition)
            .map(|symbol| symbol.address)
    }

    /// Publishes a loaded image and all its symbols together.
    ///
    /// All duplicate and state checks run before any index or image changes.
    /// On error the unpublished image is dropped and the existing executable
    /// is unchanged.
    ///
    /// # Errors
    /// Returns `Frozen`, `ProgramAlreadyLoaded`, `AlreadyDefined`, or
    /// `InvalidObject` for a non-definition.
    pub fn commit(
        &mut self,
        program: bool,
        object: LoadedObject<M>,
        definitions: Vec<Definition>,
    ) -> Result<(), Error> {
        self.can_load(program)?;
        let mut staged = HashSet::new();
        let mut staged_ids = HashSet::new();
        for definition in &definitions {
            if !definition.symbol.is_definition {
                return Err(Error::InvalidObject);
            }
            let key = (definition.symbol.full_name.as_str(), definition.scope);
            if self
                .symbol_id(&definition.symbol.full_name, definition.scope)
                .is_some()
                || self.symbols.contains_key(&definition.id)
                || !staged.insert(key)
                || !staged_ids.insert(definition.id)
            {
                return Err(Error::AlreadyDefined);
            }
        }
        for definition in definitions {
            self.names
                .entry(definition.scope)
                .or_default()
                .insert(definition.symbol.full_name.clone(), definition.id);
            self.symbols.insert(definition.id, definition.symbol);
            self.symbol_ids.push(definition.id);
        }
        self.program_loaded |= program;
        self.loaded.push(object);
        Ok(())
    }

    /// Publishes one frontend-owned external variable definition.
    ///
    /// # Errors
    /// Returns `Frozen`, `AlreadyDefined`, or `InvalidObject` for a
    /// non-definition without changing the executable.
    pub fn define_external(&mut self, definition: Definition) -> Result<(), Error> {
        if self.frozen {
            return Err(Error::Frozen);
        }
        if !definition.symbol.is_definition {
            return Err(Error::InvalidObject);
        }
        if self
            .symbol_id(&definition.symbol.full_name, definition.scope)
            .is_some()
            || self.symbols.contains_key(&definition.id)
        {
            return Err(Error::AlreadyDefined);
        }
        self.names
            .entry(definition.scope)
            .or_default()
            .insert(definition.symbol.full_name.clone(), definition.id);
        self.symbols.insert(definition.id, definition.symbol);
        self.symbol_ids.push(definition.id);
        Ok(())
    }

    /// Resolves a device range in any loaded image to a host mapping.
    ///
    /// # Errors
    /// Returns `InvalidRange` for a range that starts in an image but extends
    /// beyond it.
    pub fn host_range(&self, address: u64, size: usize) -> Result<Option<usize>, Error> {
        for object in &self.loaded {
            if let Some(host) = object.image.host_range(address, size)? {
                return Ok(Some(host));
            }
        }
        Ok(None)
    }
}
