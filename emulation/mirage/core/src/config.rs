use serde::{Deserialize, Serialize};

use crate::common::{SimpleType, SimpleValue};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OptionDef {
    pub name: String,
    pub dtype: SimpleType,
    pub description: String,
    pub default: Option<SimpleValue>,
}
