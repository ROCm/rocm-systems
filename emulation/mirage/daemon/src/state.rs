//! Shared state plumbed through every axum handler.

use std::sync::Arc;

use mirage_supervisor::SessionManager;

/// State shared by the HTTP handlers.
///
/// It holds the *same* [`SessionManager`] the Unix-socket control plane
/// serves, not a second view of the world. That is the point of hosting
/// both surfaces in one process: the dashboard and the CLI see one set of
/// sessions, and neither can observe state the other cannot.
#[derive(Clone, Debug)]
pub struct AppState {
    /// The supervisor every request is answered from.
    pub ctl: Arc<SessionManager>,
}

impl AppState {
    /// Wrap a manager for the HTTP layer.
    #[must_use]
    pub fn new(manager: Arc<SessionManager>) -> Self {
        Self { ctl: manager }
    }
}
