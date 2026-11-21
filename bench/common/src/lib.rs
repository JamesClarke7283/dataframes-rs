//! Shared support for the comparison benchmarks.
//!
//! Every library under test gets its own driver binary so that allocators,
//! thread pools and CPU caches of one library never interfere with another.
//! What they share lives here:
//!
//! * [`Dataset`] — the input data, generated from a fixed seed so every driver
//!   (and the Python driver, which re-implements the same generator) sees
//!   bit-identical columns;
//! * [`Suite`] — the timing loop and the JSON-lines result format;
//! * the case names, so a typo cannot silently drop a comparison.
//!
//! A case also reports a *check value* (a row count, a sum, ...) computed from
//! its result. The report script refuses to compare timings whose check values
//! disagree, which keeps the drivers honest about doing the same work.

use std::fmt::Write as _;
use std::hint::black_box;
use std::io::Write as _;
use std::time::{Duration, Instant};

/// xoshiro256++ seeded through SplitMix64. Mirrored in `bench/python/datagen.py`.
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0].wrapping_add(self.s[3]).rotate_left(23).wrapping_add(self.s[0]);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform in `0..bound` (multiply-shift; `bound` must fit in 32 bits).
    #[inline]
    pub fn below(&mut self, bound: u64) -> u64 {
        ((self.next_u64() >> 32) * bound) >> 32
    }

    /// Uniform in `[0, 1)` with 53 random bits.
    #[inline]
    pub fn unit_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Number of distinct values in the string column.
pub const VOCAB: usize = 1000;
/// Number of distinct group keys.
pub const GROUPS: u64 = 1000;
/// Number of random point look-ups per `get_random` iteration.
pub const LOOKUPS: usize = 1000;

/// The benchmark table. One row per index; columns are independent.
pub struct Dataset {
    pub rows: usize,
    /// `0..rows`, sorted.
    pub id: Vec<i64>,
    /// Uniform in `0..1000`: `a < 10 * p` selects `p` percent of the rows.
    pub a: Vec<i64>,
    /// Uniform in `[0, 1)`.
    pub b: Vec<f64>,
    /// Uniform in `0..100`.
    pub c: Vec<i32>,
    /// Uniform in `[0, 1)`.
    pub d: Vec<f32>,
    /// Uniform in `0..1_000_000`.
    pub e: Vec<i64>,
    /// Group key, uniform in `0..GROUPS`.
    pub k: Vec<i32>,
    /// Index into [`vocab`] for the string column.
    pub s: Vec<u32>,
    /// Random row indices for point look-ups.
    pub lookups: Vec<u32>,
}

impl Dataset {
    pub fn generate(rows: usize) -> Self {
        let mut rng = Rng::new(0xDA7A_F7A3 ^ rows as u64);
        let id = (0..rows as i64).collect();
        let a = (0..rows).map(|_| rng.below(1000) as i64).collect();
        let b = (0..rows).map(|_| rng.unit_f64()).collect();
        let c = (0..rows).map(|_| rng.below(100) as i32).collect();
        let d = (0..rows).map(|_| rng.unit_f64() as f32).collect();
        let e = (0..rows).map(|_| rng.below(1_000_000) as i64).collect();
        let k = (0..rows).map(|_| rng.below(GROUPS) as i32).collect();
        let s = (0..rows).map(|_| rng.below(VOCAB as u64) as u32).collect();
        let lookups = (0..LOOKUPS).map(|_| rng.below(rows.max(1) as u64) as u32).collect();
        Dataset { rows, id, a, b, c, d, e, k, s, lookups }
    }
}

/// The string vocabulary: word `i` is `"w"` followed by `i` and a tail whose
/// length depends on `i`, so lengths range from 2 to 24 bytes and straddle the
/// 12-byte inline limit of string views.
pub fn vocab() -> Vec<String> {
    (0..VOCAB)
        .map(|i| {
            let mut w = format!("w{i}");
            for j in 0..(i * 7 % 20) {
                w.push((b'a' + ((i + j) % 26) as u8) as char);
            }
            w
        })
        .collect()
}

/// Command-line options shared by every driver.
#[derive(Clone, Debug)]
pub struct Opts {
    pub rows: usize,
    pub filter: Option<String>,
    pub min_time: Duration,
    pub out: Option<String>,
    pub list: bool,
}

impl Opts {
    /// `--rows N  --case SUBSTR  --time-ms T  --out FILE  --list`
    pub fn from_args() -> Self {
        let mut o = Opts { rows: 1_000_000, filter: None, min_time: Duration::from_millis(300), out: None, list: false };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut value = |name: &str| args.next().unwrap_or_else(|| panic!("{name} needs a value"));
            match a.as_str() {
                "--rows" => o.rows = value("--rows").replace('_', "").parse().expect("--rows takes an integer"),
                "--case" => o.filter = Some(value("--case")),
                "--time-ms" => o.min_time = Duration::from_millis(value("--time-ms").parse().expect("--time-ms takes an integer")),
                "--out" => o.out = Some(value("--out")),
                "--list" => o.list = true,
                other => panic!("unknown argument {other:?}"),
            }
        }
        o
    }
}

/// One measured case.
#[derive(Clone, Debug)]
pub struct Record {
    pub lib: String,
    pub case: String,
    pub rows: usize,
    pub median_ns: f64,
    pub min_ns: f64,
    pub mean_ns: f64,
    pub samples: usize,
    pub check: f64,
}

/// Collects [`Record`]s for one library.
pub struct Suite {
    lib: String,
    opts: Opts,
    records: Vec<Record>,
}

impl Suite {
    pub fn new(lib: &str, opts: &Opts) -> Self {
        Suite { lib: lib.to_string(), opts: opts.clone(), records: Vec::new() }
    }

    pub fn rows(&self) -> usize {
        self.opts.rows
    }

    /// Time `f`. `check` maps one result to a number that must agree across
    /// libraries (within floating-point tolerance).
    ///
    /// The result of `f` is dropped inside the timed region: releasing the
    /// output is part of the cost of producing it in a loop.
    pub fn bench<R>(&mut self, case: &str, mut f: impl FnMut() -> R, check: impl FnOnce(R) -> f64) {
        if let Some(filter) = &self.opts.filter {
            // `--case a,b` selects by substring; `--case =name` selects the
            // case called `name` together with its `name@variant` forms.
            let base = case.split('@').next().unwrap_or(case);
            let selected = filter.split(',').any(|pat| match pat.strip_prefix('=') {
                Some(exact) => base == exact,
                None => case.contains(pat),
            });
            if !selected {
                return;
            }
        }
        if self.opts.list {
            println!("{case}");
            return;
        }
        let check = check(f());

        // Warm up: caches, branch predictors, allocator pools, lazy statics.
        let warm = (self.opts.min_time / 4).max(Duration::from_millis(20));
        let start = Instant::now();
        let mut iters = 0u64;
        while start.elapsed() < warm || iters < 3 {
            black_box(f());
            iters += 1;
        }
        let per_iter = start.elapsed().as_nanos() as f64 / iters as f64;
        // Batch very fast operations so that one sample is at least ~20 µs.
        let batch = ((20_000.0 / per_iter.max(0.01)).ceil() as u64).clamp(1, 1_000_000);

        let mut samples: Vec<f64> = Vec::new();
        let start = Instant::now();
        while (start.elapsed() < self.opts.min_time || samples.len() < 15) && samples.len() < 20_000 {
            let t = Instant::now();
            for _ in 0..batch {
                black_box(f());
            }
            samples.push(t.elapsed().as_nanos() as f64 / batch as f64);
        }
        samples.sort_by(|a, b| a.total_cmp(b));
        let record = Record {
            lib: self.lib.clone(),
            case: case.to_string(),
            rows: self.opts.rows,
            median_ns: samples[samples.len() / 2],
            min_ns: samples[0],
            mean_ns: samples.iter().sum::<f64>() / samples.len() as f64,
            samples: samples.len(),
            check,
        };
        eprintln!(
            "{:<12} {:<28} rows={:<9} median={:>12}  min={:>12}  n={}",
            record.lib,
            record.case,
            record.rows,
            fmt_ns(record.median_ns),
            fmt_ns(record.min_ns),
            record.samples
        );
        self.records.push(record);
    }

    /// Append the records as JSON lines to `--out` (or print them to stdout).
    pub fn finish(self) {
        let mut text = String::new();
        for r in &self.records {
            writeln!(
                text,
                "{{\"lib\":\"{}\",\"case\":\"{}\",\"rows\":{},\"median_ns\":{:.3},\"min_ns\":{:.3},\"mean_ns\":{:.3},\"samples\":{},\"check\":{:e}}}",
                r.lib, r.case, r.rows, r.median_ns, r.min_ns, r.mean_ns, r.samples, r.check
            )
            .expect("writing to a String cannot fail");
        }
        match &self.opts.out {
            Some(path) => {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .unwrap_or_else(|e| panic!("cannot open {path}: {e}"));
                file.write_all(text.as_bytes()).expect("cannot write results");
            }
            None => print!("{text}"),
        }
    }
}

/// Human-readable duration.
pub fn fmt_ns(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{ns:.1} ns")
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.3} ms", ns / 1_000_000.0)
    } else {
        format!("{:.3} s", ns / 1_000_000_000.0)
    }
}
