//! What the remap prologue does in one replay (the control block's mode word), and what that replay left in the
//! tool buffers.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Hardware ids, no markers: the prologue's fixed cost only.
    Off,
    /// Hardware ids; log every workgroup's location and turn.
    Record,
    /// Ids from the table; log placement.
    Table,
    /// Hardware ids with A1 identity markers.
    MarkNatural,
    /// Ids from the table with A1 identity markers (what ATT profiles).
    MarkTable,
    /// Sparse launches only: log every workgroup's location, then end it before the body.
    Probe,
}

impl Mode {
    pub const ALL: [Mode; 6] = [
        Mode::Off,
        Mode::Record,
        Mode::Table,
        Mode::MarkNatural,
        Mode::MarkTable,
        Mode::Probe,
    ];

    pub fn code(self) -> u32 {
        self as u32
    }

    pub fn name(self) -> &'static str {
        match self {
            Mode::Off => "off",
            Mode::Record => "record",
            Mode::Table => "table",
            Mode::MarkNatural => "mark_natural",
            Mode::MarkTable => "mark_table",
            Mode::Probe => "probe",
        }
    }

    pub fn parse(s: &str) -> Result<Self> {
        let s = s.replace('-', "_");
        Mode::ALL
            .into_iter()
            .find(|m| m.name() == s)
            .with_context(|| {
                format!("unknown remap mode '{s}' (off, record, table, mark_natural, mark_table, probe)")
            })
    }

    pub fn uses_table(self) -> bool {
        matches!(self, Mode::Table | Mode::MarkTable)
    }

    pub fn logs(self) -> bool {
        matches!(self, Mode::Record | Mode::Table | Mode::Probe)
    }

    /// Golden verification applies: the body runs. Probe replays end every workgroup before it.
    pub fn verifies(self) -> bool {
        self != Mode::Probe
    }
}

/// What one replay left in the tool buffers; written as `outcome.json` next to the logs.
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Outcome {
    pub overflow: u32,
    pub logged: u32,
    pub expected_log_records: u32,
    /// Golden verification of this replay's outputs, when the replayer knew it.
    #[serde(default)]
    pub golden_ok: Option<bool>,
}

impl Outcome {
    pub fn valid(&self, mode: Mode) -> bool {
        self.overflow == 0 && (!mode.logs() || self.logged == self.expected_log_records)
    }
    /// A workgroup found no table entry: placement drifted from the recorded one, so the replay ran some tile twice
    /// and is expected to mismatch. Drifted replays are discarded and replaced, never accepted.
    pub fn drifted(&self) -> bool {
        self.overflow > 0
    }
}
