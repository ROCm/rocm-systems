use serde::{Deserialize, Serialize};

use crate::common::{SimpleType, SimpleValue};

/// The section a rocjitsu simulation config uses to describe a DBT guest.
pub const DBT_GUEST_SECTION: &str = "dbt_guest";

/// The `dbt_guest` block of `doc`, if `doc` asks for a guest at all.
///
/// The section alone does not ask for one. `enabled` is what the
/// emulator reads, and a block with it off is an ordinary config that
/// happens to carry settings for a guest it is not using. The two have
/// to be told apart in the same way everywhere they are told apart: the
/// front end picks the execution mode from this, the backend builds the
/// overlay from it, and when they disagree a config runs in a mode the
/// backend then refuses.
#[must_use]
pub fn enabled_dbt_guest(doc: &serde_json::Value) -> Option<&serde_json::Value> {
    let guest = doc.get(DBT_GUEST_SECTION)?;
    guest
        .get("enabled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
        .then_some(guest)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionDef {
    pub name: String,
    pub dtype: SimpleType,
    pub description: String,
    pub default: Option<SimpleValue>,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    /// Everything short of `enabled: true` is not a guest.
    ///
    /// Absent, absent-key, `false`, and a non-boolean all mean the same
    /// thing to the emulator, so they have to mean it here too.
    #[test]
    fn only_an_enabled_block_is_a_guest() {
        let guest = |body: &str| {
            let doc: serde_json::Value = serde_json::from_str(body).unwrap();
            enabled_dbt_guest(&doc).is_some()
        };
        assert!(!guest(r#"{"vm": {}, "topology": {}}"#));
        assert!(!guest(r#"{"dbt_guest": {"guest_isa": "gfx950"}}"#));
        assert!(!guest(r#"{"dbt_guest": {"enabled": false}}"#));
        assert!(!guest(r#"{"dbt_guest": {"enabled": "true"}}"#));
        assert!(guest(r#"{"dbt_guest": {"enabled": true}}"#));
    }

    /// A `dbt_guest` that is not a block at all is not an enabled one.
    ///
    /// `null`, a number, a string, an array: `Value::get` answers `None`
    /// for the `enabled` of every one of them, which is the same answer
    /// it gives for an object that omits it. Written down because the
    /// question this predicate settles — whether a config describes a
    /// guest — is asked of documents nobody validated first, and "the
    /// key is there" was once the whole of the answer.
    #[test]
    fn a_dbt_guest_that_is_not_an_object_is_not_a_guest() {
        let guest = |body: &str| {
            let doc: serde_json::Value = serde_json::from_str(body).unwrap();
            enabled_dbt_guest(&doc).is_some()
        };
        assert!(!guest(r#"{"dbt_guest": null}"#));
        assert!(!guest(r#"{"dbt_guest": 5}"#));
        assert!(!guest(r#"{"dbt_guest": "enabled"}"#));
        assert!(!guest(r#"{"dbt_guest": []}"#));
        assert!(!guest(r#"{"dbt_guest": true}"#));
    }

    /// The block comes back, not just the verdict: the backend reads the
    /// rest of it and should not have to find it a second time.
    #[test]
    fn an_enabled_block_is_returned_whole() {
        let doc: serde_json::Value =
            serde_json::from_str(r#"{"dbt_guest": {"enabled": true, "guest_isa": "gfx950"}}"#)
                .unwrap();
        let guest = enabled_dbt_guest(&doc).expect("enabled");
        assert_eq!(
            guest.get("guest_isa").and_then(|v| v.as_str()),
            Some("gfx950")
        );
    }
}
