//! Micro-timings of small operations (development tool).
use std::hint::black_box;
use std::time::Instant;

use dataframes::memory::Buffer;
use dataframes::prelude::*;

fn time<R>(name: &str, mut f: impl FnMut() -> R) {
    let iters = 2_000_000u32;
    for _ in 0..10_000 {
        black_box(f());
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..iters {
            black_box(f());
        }
        best = best.min(t.elapsed().as_nanos() as f64 / iters as f64);
    }
    println!("{name:<44} {best:>8.1} ns");
}

fn main() {
    let n = 1_000_000usize;
    let v: Vec<i64> = (0..n as i64).collect();
    let buf = Buffer::from_slice(&v);
    let ca = Int64Chunked::from_slice("a".into(), &v);
    let s = ca.clone().into_series();
    let cols: Vec<Series> = (0..6).map(|i| Int64Chunked::from_slice(format!("c{i}").into(), &v).into_series()).collect();
    let df = DataFrame::new(n, cols).unwrap();
    println!("size_of: Buffer={} Int64Chunked={} Series={} DataFrame={}", size_of::<Buffer<i64>>(), size_of::<Int64Chunked>(), size_of::<Series>(), size_of::<DataFrame>());
    time("Buffer::slice + drop", || buf.slice(black_box(1000), 500_000));
    time("Buffer::clone + drop", || buf.clone());
    time("Int64Chunked::slice + drop", || ca.slice(black_box(1000), 500_000));
    time("Int64Chunked::clone + drop", || ca.clone());
    time("Series::slice + drop", || s.slice(black_box(1000), 500_000));
    time("Series::clone + drop", || s.clone());
    time("DataFrame::slice (6 cols) + drop", || df.slice(black_box(1000), 500_000));
    time("DataFrame::clone (6 cols) + drop", || df.clone());
    time("DataFrame::head(10) + drop", || df.head(Some(10)));
    time("Vec<u64>::with_capacity(96) + drop", || Vec::<u64>::with_capacity(black_box(96)));
    time("PlSmallStr clone", || black_box(ca.name()).clone());
    time("Int64Chunked::get", || ca.get(black_box(777_777)));
    time("df.column(\"c5\")", || df.column(black_box("c5")).unwrap().len());
}
