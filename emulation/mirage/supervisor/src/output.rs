//! The output fan-out for an attached exec.
//!
//! An exec's output has to reach an unknown number of clients that may
//! attach at any point in its life — before it starts, halfway through,
//! or after it has already exited. [`OutputHub`] is what makes all three
//! well defined:
//!
//! * a bounded **replay buffer** holds recent output, so a late attacher
//!   sees what it missed instead of an empty stream;
//! * a **broadcast channel** carries live output to everyone currently
//!   attached;
//! * the two are updated under one lock, so a client subscribing
//!   concurrently with a write can neither miss a packet nor see it
//!   twice.
//!
//! The buffer is bounded in *bytes*, not packets. A workload that writes
//! gigabytes must not be able to exhaust the daemon's memory just because
//! nobody is attached — which, with one long-lived daemon owning every
//! session, would take out unrelated sessions too.

use std::collections::VecDeque;
use std::sync::Mutex;

use mirage_core::ctl::StreamPacket;
use tokio::sync::broadcast;

/// Default cap on retained output per exec, in bytes.
pub const DEFAULT_REPLAY_BYTES: usize = 1024 * 1024;

/// Capacity of the live broadcast channel, in packets.
///
/// A slow client that falls this far behind is disconnected by the
/// channel rather than being allowed to stall the writer. Attach clients
/// are interactive and drain promptly; the depth only has to absorb a
/// burst.
const BROADCAST_CAPACITY: usize = 1024;

/// Fan-out for one exec's output.
#[derive(Debug)]
pub struct OutputHub {
    live: broadcast::Sender<StreamPacket>,
    replay: Mutex<Replay>,
    max_bytes: usize,
}

#[derive(Debug, Default)]
struct Replay {
    packets: VecDeque<StreamPacket>,
    bytes: usize,
    /// How many bytes have been dropped off the front to stay in budget.
    dropped_bytes: u64,
    /// Set once the exec has finished; attaching after this replays and
    /// ends immediately rather than waiting for live output.
    finished: Option<i32>,
}

/// What a new subscriber receives: everything retained so far, plus a
/// live feed of what comes next.
#[derive(Debug)]
pub struct Subscription {
    /// Retained packets, oldest first.
    pub replay: Vec<StreamPacket>,
    /// The live feed. `None` when the exec had already finished, in which
    /// case `replay` is the complete remaining story.
    pub live: Option<broadcast::Receiver<StreamPacket>>,
    /// Exit code, if the exec has already finished.
    pub finished: Option<i32>,
    /// Bytes discarded from the front of the replay buffer to stay within
    /// the cap. Non-zero means the replay is truncated.
    pub dropped_bytes: u64,
}

impl OutputHub {
    /// Build a hub retaining at most `max_bytes` of output.
    #[must_use]
    pub fn new(max_bytes: usize) -> Self {
        let (live, _) = broadcast::channel(BROADCAST_CAPACITY);
        Self {
            live,
            replay: Mutex::new(Replay::default()),
            max_bytes,
        }
    }

    /// Publish a packet to the replay buffer and every live subscriber.
    ///
    /// Ordering with [`OutputHub::subscribe`] is what makes the hub
    /// correct: both take the same lock, and this one appends to the
    /// buffer *before* broadcasting. A subscriber that runs between the
    /// two therefore sees the packet in its replay and not on its live
    /// feed; one that runs after sees it only on the feed. Neither can
    /// see it twice or lose it.
    pub fn publish(&self, packet: StreamPacket) {
        let mut replay = self.lock();
        let size = packet_bytes(&packet);
        replay.packets.push_back(packet.clone());
        replay.bytes += size;
        // Evict from the front until we are back in budget, but never
        // evict the last packet. A single write larger than the whole
        // budget would otherwise be dropped the instant it arrived,
        // silently discarding output rather than truncating history —
        // and the packet most likely to be huge is the one carrying a
        // workload's final diagnostic.
        while replay.bytes > self.max_bytes && replay.packets.len() > 1 {
            let Some(dropped) = replay.packets.pop_front() else {
                break;
            };
            let n = packet_bytes(&dropped);
            replay.bytes = replay.bytes.saturating_sub(n);
            replay.dropped_bytes += n as u64;
        }
        // A send error just means nobody is attached right now.
        let _ = self.live.send(packet);
    }

    /// Mark the exec finished with `exit_code`.
    ///
    /// Publishes a final [`StreamPacket::ExecExit`] and records the code
    /// so that later subscribers terminate immediately instead of waiting
    /// on a feed that will never produce anything again.
    pub fn finish(&self, exit_code: i32) {
        {
            let mut replay = self.lock();
            if replay.finished.is_some() {
                // Already finished; do not publish a second exit.
                return;
            }
            replay.finished = Some(exit_code);
        }
        self.publish(StreamPacket::ExecExit { exit_code });
    }

    /// Whether the exec has finished, and with what code.
    #[must_use]
    pub fn finished(&self) -> Option<i32> {
        self.lock().finished
    }

    /// Subscribe, receiving the retained output and a live feed.
    #[must_use]
    pub fn subscribe(&self) -> Subscription {
        let replay = self.lock();
        // Subscribe while holding the lock so no packet can slip between
        // snapshotting the buffer and joining the broadcast.
        let live = replay.finished.is_none().then(|| self.live.subscribe());
        Subscription {
            replay: replay.packets.iter().cloned().collect(),
            live,
            finished: replay.finished,
            dropped_bytes: replay.dropped_bytes,
        }
    }

    /// Bytes currently retained.
    #[must_use]
    pub fn retained_bytes(&self) -> usize {
        self.lock().bytes
    }

    /// Everything retained, concatenated per stream — the `mirage logs`
    /// (non-following) view.
    #[must_use]
    pub fn snapshot(&self) -> Vec<StreamPacket> {
        self.lock().packets.iter().cloned().collect()
    }

    /// Lock the replay buffer, recovering from a poisoned mutex.
    ///
    /// Poisoning here means a panic while holding the lock. The buffer is
    /// plain data with no invariant that a panic could have left broken
    /// mid-update, and refusing to serve output for the rest of the
    /// daemon's life would be a far worse outcome than continuing.
    fn lock(&self) -> std::sync::MutexGuard<'_, Replay> {
        self.replay.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for OutputHub {
    fn default() -> Self {
        Self::new(DEFAULT_REPLAY_BYTES)
    }
}

/// Approximate retained size of a packet.
fn packet_bytes(packet: &StreamPacket) -> usize {
    match packet {
        StreamPacket::Output { data, .. } => data.len(),
        // Control frames are tiny but not free; count them so a flood of
        // them cannot grow the buffer without bound either.
        StreamPacket::NodeExit { .. } | StreamPacket::ExecExit { .. } => 32,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use mirage_core::ctl::StdStream;

    fn out(node: u32, data: &[u8]) -> StreamPacket {
        StreamPacket::Output {
            node,
            stream: StdStream::Stdout,
            data: data.to_vec(),
        }
    }

    #[test]
    fn replay_returns_everything_written_before_subscribing() {
        let hub = OutputHub::default();
        hub.publish(out(0, b"one"));
        hub.publish(out(0, b"two"));
        let sub = hub.subscribe();
        assert_eq!(sub.replay.len(), 2);
        assert_eq!(sub.dropped_bytes, 0);
        assert!(sub.live.is_some());
        assert!(sub.finished.is_none());
    }

    #[tokio::test]
    async fn live_feed_carries_output_published_after_subscribing() {
        let hub = OutputHub::default();
        let mut sub = hub.subscribe();
        hub.publish(out(0, b"later"));
        let live = sub.live.as_mut().unwrap();
        assert_eq!(live.recv().await.unwrap(), out(0, b"later"));
    }

    #[tokio::test]
    async fn a_packet_is_never_both_replayed_and_broadcast() {
        // The exact race the lock ordering exists to prevent: a client
        // subscribing while output is being written must see each packet
        // exactly once across (replay ++ live).
        let hub = std::sync::Arc::new(OutputHub::default());
        let writer = {
            let hub = hub.clone();
            std::thread::spawn(move || {
                for i in 0..500u32 {
                    hub.publish(out(0, i.to_string().as_bytes()));
                }
            })
        };
        // Subscribe at an arbitrary point mid-flight.
        std::thread::sleep(std::time::Duration::from_micros(200));
        let mut sub = hub.subscribe();
        writer.join().unwrap();
        hub.finish(0);

        let mut seen: Vec<String> = sub
            .replay
            .iter()
            .filter_map(payload)
            .collect::<Vec<_>>();
        if let Some(live) = sub.live.as_mut() {
            while let Ok(pkt) = live.try_recv() {
                if let Some(p) = payload(&pkt) {
                    seen.push(p);
                }
            }
        }
        let unique: std::collections::HashSet<&String> = seen.iter().collect();
        assert_eq!(
            unique.len(),
            seen.len(),
            "a packet was delivered twice: {seen:?}"
        );
        let expected: Vec<String> = (0..500u32).map(|i| i.to_string()).collect();
        assert_eq!(seen, expected, "packets were lost or reordered");
    }

    fn payload(pkt: &StreamPacket) -> Option<String> {
        match pkt {
            StreamPacket::Output { data, .. } => {
                Some(String::from_utf8_lossy(data).into_owned())
            }
            _ => None,
        }
    }

    #[test]
    fn replay_is_bounded_and_reports_what_it_dropped() {
        let hub = OutputHub::new(100);
        for _ in 0..50 {
            hub.publish(out(0, &[b'x'; 10]));
        }
        assert!(
            hub.retained_bytes() <= 100,
            "retained {} bytes, cap is 100",
            hub.retained_bytes()
        );
        let sub = hub.subscribe();
        assert!(
            sub.dropped_bytes > 0,
            "truncation must be visible to the client"
        );
    }

    #[test]
    fn a_single_packet_larger_than_the_cap_is_still_retained() {
        // Dropping it entirely would lose output with no way to notice;
        // the buffer over-shoots for one packet rather than swallowing it.
        let hub = OutputHub::new(10);
        hub.publish(out(0, &[b'x'; 1000]));
        assert_eq!(hub.subscribe().replay.len(), 1);

        // And a following packet evicts it rather than accumulating.
        hub.publish(out(0, b"next"));
        let sub = hub.subscribe();
        assert_eq!(sub.replay.len(), 1);
        assert_eq!(sub.dropped_bytes, 1000);
    }

    #[test]
    fn subscribing_after_the_exec_finished_replays_and_ends() {
        let hub = OutputHub::default();
        hub.publish(out(0, b"done"));
        hub.finish(3);

        let sub = hub.subscribe();
        assert_eq!(sub.finished, Some(3));
        assert!(
            sub.live.is_none(),
            "a finished exec must not hand out a live feed that never yields"
        );
        assert!(matches!(
            sub.replay.last(),
            Some(StreamPacket::ExecExit { exit_code: 3 })
        ));
    }

    #[test]
    fn finish_is_idempotent() {
        let hub = OutputHub::default();
        hub.finish(0);
        hub.finish(9);
        assert_eq!(hub.finished(), Some(0));
        let exits = hub
            .snapshot()
            .into_iter()
            .filter(|p| matches!(p, StreamPacket::ExecExit { .. }))
            .count();
        assert_eq!(exits, 1, "exactly one ExecExit must ever be published");
    }

    #[test]
    fn publishing_with_no_subscribers_is_fine() {
        let hub = OutputHub::default();
        for _ in 0..1000 {
            hub.publish(out(0, b"nobody listening"));
        }
        assert!(hub.retained_bytes() > 0);
    }
}
