//! `attach` implements the streaming "tail -f" for an exec.
//!
//! When the CLI calls `MirageCtl::session_attach`, this module returns
//! a `Stream<Item = StreamPacket>` that:
//!
//! * tails each node's `stdout` and `stderr` files
//! * emits `NodeExit` once a node's `exit_code` file appears
//! * emits a final `ExecExit` once the overall `status.json` shows
//!   `ended: true`
//! * closes when the exec is fully done (or after a grace period if
//!   the exec directory is removed).

use std::collections::BTreeMap;
use std::time::Duration;

use futures::stream::Stream;
use tokio::time::interval;
use tokio_stream::wrappers::ReceiverStream;

use crate::ctl::{StdStream, StreamPacket, StreamPacketStream};
use crate::paths::ExecLayout;

pub fn attach_stream(layout: ExecLayout) -> StreamPacketStream {
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(async move {
        let mut tails: BTreeMap<(u32, StdStream), u64> = BTreeMap::new();
        let mut emitted_node_exit: BTreeMap<u32, bool> = BTreeMap::new();
        let mut tick = interval(Duration::from_millis(50));
        loop {
            tick.tick().await;

            // discover nodes
            let node_root = layout.node_root();
            let mut nodes: Vec<u32> = Vec::new();
            if let Ok(rd) = std::fs::read_dir(&node_root) {
                for e in rd.flatten() {
                    if let Some(s) = e.file_name().to_str()
                        && let Ok(n) = s.parse::<u32>()
                    {
                        nodes.push(n);
                    }
                }
            }
            nodes.sort();

            for n in &nodes {
                let nl = layout.node(*n);
                for (st, path) in [
                    (StdStream::Stdout, nl.stdout()),
                    (StdStream::Stderr, nl.stderr()),
                ] {
                    if let Ok(meta) = std::fs::metadata(&path) {
                        let len = meta.len();
                        let cursor = tails.entry((*n, st)).or_insert(0);
                        if len > *cursor {
                            let to_read = len - *cursor;
                            if let Ok(mut f) = std::fs::File::open(&path) {
                                use std::io::{Read, Seek, SeekFrom};
                                if f.seek(SeekFrom::Start(*cursor)).is_ok() {
                                    let mut buf = vec![0u8; to_read as usize];
                                    if let Ok(read) = f.read(&mut buf) {
                                        buf.truncate(read);
                                        *cursor += read as u64;
                                        if !buf.is_empty()
                                            && tx
                                                .send(StreamPacket::Output {
                                                    node: *n,
                                                    stream: st,
                                                    data: buf,
                                                })
                                                .await
                                                .is_err()
                                        {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // node exit?
                if !emitted_node_exit.get(n).copied().unwrap_or(false)
                    && let Ok(Some(s)) = crate::state::read_small_str(&nl.exit_code())
                    && let Ok(code) = s.parse::<i32>()
                {
                    emitted_node_exit.insert(*n, true);
                    if tx
                        .send(StreamPacket::NodeExit {
                            node: *n,
                            exit_code: code,
                        })
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
            }

            // overall finished?
            if let Ok(Some(status)) =
                crate::state::read_json_opt::<crate::exec::ExecStatus>(&layout.status())
                && status.ended
            {
                // one last pass over outputs in case anything was written
                // between the last poll and the exit code being written
                // is handled by the cursor logic above; just exit.
                let code = status.exit_code.unwrap_or(0);
                let _ = tx.send(StreamPacket::ExecExit { exit_code: code }).await;
                return;
            }

            // if the layout disappeared, exit.
            if !layout.root.exists() {
                let _ = tx.send(StreamPacket::ExecExit { exit_code: -1 }).await;
                return;
            }
        }
    });
    Box::pin(ReceiverStream::new(rx)) as Pin<Box<dyn Stream<Item = StreamPacket> + Send>>
}

use std::pin::Pin;
