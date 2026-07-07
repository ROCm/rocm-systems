//! `mirage_dashboard`: the bundled web UI single-page app.
//!
//! The compiled SPA assets are embedded into the binary behind the
//! `webui` feature. Without that feature this crate is empty and its
//! `build.rs` is a no-op, so building mirage never requires Node.js.

#[cfg(feature = "webui")]
pub mod spa {
    include!(concat!(env!("OUT_DIR"), "/dashboard_assets.rs"));
}

#[cfg(all(test, feature = "webui"))]
mod tests {
    use super::*;

    #[test]
    fn embeds_dashboard_index() {
        let index = spa::get("/").expect("dashboard index should be embedded");

        assert_eq!(index.path, "index.html");
        assert_eq!(index.content_type, "text/html; charset=utf-8");
        assert!(!index.bytes.is_empty());
    }

    #[test]
    fn spa_routes_fall_back_to_index() {
        let index = spa::get("index.html").expect("dashboard index should be embedded");
        let route = spa::get_spa("/simulators/custom-route")
            .expect("unknown dashboard routes should fall back to the SPA index");

        assert_eq!(route.bytes, index.bytes);
    }
}
