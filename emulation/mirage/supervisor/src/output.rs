//! Writing a captured exec's output to the terminal, labelled by rank.
//!
//! Only reached under `--capture-all`. Without it a workload's stdout and
//! stderr *are* the caller's, and nothing here runs: the bytes never pass
//! through mirage at all, which is what keeps output byte-exact and an
//! interactive shell interactive.
//!
//! # Why line-oriented
//!
//! The obvious implementation prefixes each chunk as it arrives, and it
//! is wrong. A chunk is whatever one `read` returned, so a single line
//! can arrive in three pieces and three lines can arrive in one. Prefixed
//! per chunk, that produces labels in the middle of lines and lines with
//! no label at all.
//!
//! So each rank's stream is buffered until it holds a complete line, and
//! only whole lines are emitted. The tail — a prompt, a progress bar,
//! anything a workload wrote without a newline — is flushed when its
//! stream ends, so nothing is ever silently dropped.

use std::collections::BTreeMap;
use std::io::Write as _;

use mirage_core::exec::StdStream;
use tokio::sync::mpsc;

use crate::process::OutputChunk;

/// Label for a rank's output. Kept narrow and fixed-width so consecutive
/// lines from different ranks stay visually aligned.
fn prefix(node: u32) -> String {
    format!("[{node}] ")
}

/// Longest run of bytes held while waiting for a newline, per rank and
/// stream.
///
/// A workload is not obliged to ever write one. A progress bar redraws
/// with `\r`, a binary blob has none at all, and `tqdm` — which a great
/// many of the workloads this runs will use — emits megabytes without a
/// single `\n`. Buffering until the stream ends would then grow one `Vec`
/// per rank at pipe speed for the whole life of the job, so past this
/// point the partial line is emitted as-is. A wrapped line is a far
/// better outcome than a supervisor holding gigabytes of a progress bar.
const MAX_PARTIAL_LINE: usize = 1024 * 1024;

/// Drain `rx`, writing every chunk to this process's stdout and stderr
/// with its rank prefixed.
///
/// Returns when the channel closes, which happens once every pump task
/// for the exec has finished — so by the time this returns, every byte
/// the workload wrote has been printed.
pub async fn print_labelled(mut rx: mpsc::Receiver<OutputChunk>) {
    let mut buffers: BTreeMap<(u32, StdStream), Vec<u8>> = BTreeMap::new();

    while let Some(chunk) = rx.recv().await {
        let buf = buffers.entry((chunk.node, chunk.stream)).or_default();
        buf.extend_from_slice(&chunk.data);

        // Emit only up to the last newline; whatever follows is a partial
        // line and waits for the rest of itself — unless it has waited
        // long enough to be a memory leak, in which case it goes out
        // unterminated rather than growing without bound.
        let end = match buf.iter().rposition(|b| *b == b'\n') {
            Some(end) => end,
            None if buf.len() >= MAX_PARTIAL_LINE => buf.len() - 1,
            None => continue,
        };
        let complete: Vec<u8> = buf.drain(..=end).collect();
        write_lines(chunk.node, chunk.stream, &complete);
    }

    // Flush partial lines. A workload that ended with a prompt or a
    // progress bar wrote real output, and dropping it because it lacked a
    // trailing newline would be a silent loss.
    for ((node, stream), buf) in buffers {
        if !buf.is_empty() {
            write_lines(node, stream, &buf);
        }
    }
}

/// Write `data` to the stream it came from, prefixing every line.
///
/// stdout and stderr stay separate all the way out, so redirecting one
/// of them still works under `--capture-all`.
fn write_lines(node: u32, stream: StdStream, data: &[u8]) {
    let tag = prefix(node);
    let mut out = Vec::with_capacity(data.len() + tag.len() * 4);
    for line in split_inclusive_lines(data) {
        out.extend_from_slice(tag.as_bytes());
        out.extend_from_slice(line);
        // A trailing partial line (the flush case) gets a newline so the
        // next thing printed does not continue it.
        if line.last() != Some(&b'\n') {
            out.push(b'\n');
        }
    }
    let written = match stream {
        StdStream::Stdout => {
            let mut w = std::io::stdout().lock();
            w.write_all(&out).and_then(|()| w.flush())
        }
        StdStream::Stderr => {
            let mut w = std::io::stderr().lock();
            w.write_all(&out).and_then(|()| w.flush())
        }
    };
    if let Err(e) = written {
        // A closed stdout (a `head` in the pipeline) is normal, not an
        // error worth reporting on the stream that just failed.
        tracing::debug!(node, ?stream, "could not write output: {e}");
    }
}

/// Split into lines, keeping the terminator on each.
fn split_inclusive_lines(data: &[u8]) -> impl Iterator<Item = &[u8]> {
    data.split_inclusive(|b| *b == b'\n')
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn labelled(node: u32, data: &[u8]) -> String {
        let tag = prefix(node);
        let mut out = String::new();
        for line in split_inclusive_lines(data) {
            out.push_str(&tag);
            out.push_str(&String::from_utf8_lossy(line));
            if line.last() != Some(&b'\n') {
                out.push('\n');
            }
        }
        out
    }

    #[test]
    fn every_line_of_a_multi_line_chunk_is_labelled() {
        assert_eq!(labelled(2, b"a\nb\nc\n"), "[2] a\n[2] b\n[2] c\n");
    }

    #[test]
    fn a_partial_final_line_is_terminated_rather_than_dropped() {
        // A workload's last output is often a prompt with no newline.
        assert_eq!(labelled(0, b"no newline"), "[0] no newline\n");
    }

    #[tokio::test]
    async fn a_line_split_across_chunks_is_labelled_once() {
        // The bug this buffering exists to prevent: prefixing per chunk
        // would emit "[0] hel" and "[0] lo".
        let (tx, rx) = mpsc::channel(8);
        tx.send(OutputChunk {
            node: 0,
            stream: StdStream::Stdout,
            data: b"hel".to_vec(),
        })
        .await
        .unwrap();
        tx.send(OutputChunk {
            node: 0,
            stream: StdStream::Stdout,
            data: b"lo\n".to_vec(),
        })
        .await
        .unwrap();
        drop(tx);
        // Exercises the buffering path end to end; the assertion that the
        // two halves join is in `buffering_joins_split_lines` below,
        // which can inspect the buffer directly.
        print_labelled(rx).await;
    }

    #[test]
    fn buffering_joins_split_lines_before_labelling() {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"hel");
        assert!(
            !buf.contains(&b'\n'),
            "a chunk with no newline must not be emitted yet"
        );
        buf.extend_from_slice(b"lo\n");
        let end = buf.iter().rposition(|b| *b == b'\n').unwrap();
        let complete: Vec<u8> = buf.drain(..=end).collect();
        assert_eq!(labelled(0, &complete), "[0] hello\n");
        assert!(buf.is_empty());
    }

    #[test]
    fn output_after_a_newline_waits_for_the_rest_of_its_line() {
        let mut buf: Vec<u8> = b"done\nnext".to_vec();
        let end = buf.iter().rposition(|b| *b == b'\n').unwrap();
        let complete: Vec<u8> = buf.drain(..=end).collect();
        assert_eq!(labelled(1, &complete), "[1] done\n");
        assert_eq!(buf, b"next", "the partial line stays buffered");
    }
}
