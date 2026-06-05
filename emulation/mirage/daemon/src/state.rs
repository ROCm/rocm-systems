//! Shared `AppState` plumbed through every axum handler.

use std::sync::Arc;

use mirage_core::ctl::FileCtl;

/// Shared state.
///
/// `FileCtl` is `Send + Sync + Clone`, but we wrap it in `Arc` so
/// extensions/middleware can share it cheaply.
#[derive(Clone)]
pub struct AppState {
    pub ctl: Arc<FileCtl>,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            ctl: Arc::new(FileCtl::new()),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
