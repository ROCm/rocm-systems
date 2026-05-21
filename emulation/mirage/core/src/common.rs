#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SimpleValue {
    String(String),
    Number(f64),
    Boolean(bool),
}

pub type SimpleMap = BTreeMap<String, SimpleValue>;