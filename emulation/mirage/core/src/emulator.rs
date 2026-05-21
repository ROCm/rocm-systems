
pub struct EmulatorDef {
    pub provider: String,
    pub config: BTreeMap<String, SimpleValue>,
    
}

pub struct EmulatorConfig {
    pub debug: bool,
    pub log_level: LogLevel,
}

pub trait Emulator {
    fn run(&mut self);

    
}

