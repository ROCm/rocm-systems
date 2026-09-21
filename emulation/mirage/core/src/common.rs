use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum SimpleValue {
    String(String),
    Number(i64),
    Boolean(bool),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
pub enum SimpleType {
    String,
    Number,
    Boolean,
}

pub type SimpleMap = BTreeMap<String, SimpleValue>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Eq)]
#[serde(untagged)]
pub enum MaybeRef<T> {
    Ref(String),
    Owned(T),
}
