//! What the kernels need to know about the cache hierarchy.
//!
//! One question matters: *is this output bigger than the cache?* If it is,
//! its cache lines will be evicted before anyone reads them, so the kernels
//! write it with non-temporal stores and save the read-for-ownership of every
//! line (a third of the memory traffic of a 50 % filter). If it is not, plain
//! stores keep the result hot for the next operator.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache sizes in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CacheInfo {
    /// Per-core L2.
    pub l2: usize,
    /// Last-level cache (shared).
    pub l3: usize,
}

const FALLBACK: CacheInfo = CacheInfo { l2: 512 << 10, l3: 8 << 20 };

/// Cache sizes of the machine, detected once.
pub fn cache_info() -> CacheInfo {
    static INFO: OnceLock<CacheInfo> = OnceLock::new();
    *INFO.get_or_init(detect)
}

fn detect() -> CacheInfo {
    let mut info = FALLBACK;
    #[cfg(target_os = "linux")]
    {
        let read = |index: u32| -> Option<usize> {
            let dir = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
            let level: u32 = std::fs::read_to_string(format!("{dir}/level")).ok()?.trim().parse().ok()?;
            let kind = std::fs::read_to_string(format!("{dir}/type")).ok()?;
            if kind.trim() == "Instruction" {
                return None;
            }
            let size = std::fs::read_to_string(format!("{dir}/size")).ok()?;
            let size = size.trim();
            let (digits, unit) = size.split_at(size.find(|c: char| !c.is_ascii_digit()).unwrap_or(size.len()));
            let n: usize = digits.parse().ok()?;
            let bytes = match unit {
                "K" => n << 10,
                "M" => n << 20,
                "G" => n << 30,
                _ => n,
            };
            Some(((level as usize) << 48) | bytes)
        };
        for index in 0..8 {
            if let Some(packed) = read(index) {
                let (level, bytes) = (packed >> 48, packed & ((1 << 48) - 1));
                match level {
                    2 => info.l2 = bytes,
                    3 => info.l3 = bytes,
                    _ => {}
                }
            }
        }
    }
    info
}

const UNSET: usize = usize::MAX;
static STREAM_THRESHOLD: AtomicUsize = AtomicUsize::new(UNSET);

/// Upper bound of the default threshold. A server CPU reports a last-level
/// cache of hundreds of megabytes, but it is shared by dozens of cores (and,
/// in a virtual machine, by other tenants): a result of tens of megabytes
/// does not stay in it.
const MAX_DEFAULT_THRESHOLD: usize = 32 << 20;

/// Outputs of at least this many bytes are written with non-temporal stores.
///
/// Defaults to the size of the last-level cache, at most 32 MiB; override with
/// [`set_stream_threshold`] or the `DATAFRAMES_STREAM_BYTES` environment
/// variable (`0` streams everything, a huge value disables streaming).
#[inline]
pub fn stream_threshold() -> usize {
    match STREAM_THRESHOLD.load(Ordering::Relaxed) {
        UNSET => init_stream_threshold(),
        t => t,
    }
}

#[cold]
fn init_stream_threshold() -> usize {
    let t = std::env::var("DATAFRAMES_STREAM_BYTES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| cache_info().l3.min(MAX_DEFAULT_THRESHOLD))
        .min(UNSET - 1);
    STREAM_THRESHOLD.store(t, Ordering::Relaxed);
    t
}

/// Change [`stream_threshold`].
pub fn set_stream_threshold(bytes: usize) {
    STREAM_THRESHOLD.store(bytes.min(UNSET - 1), Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_sizes_are_plausible() {
        let info = cache_info();
        assert!(info.l2 >= 64 << 10 && info.l2 <= 64 << 20, "{info:?}");
        assert!(info.l3 >= info.l2 && info.l3 <= 4 << 30, "{info:?}");
        assert!(stream_threshold() > 0 || std::env::var("DATAFRAMES_STREAM_BYTES").is_ok());
    }
}
