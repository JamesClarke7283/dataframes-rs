//! Interleaved A/B timing of kernel strategies (development tool).
//!
//! Variants are run round-robin so that frequency scaling and thermal drift
//! hit all of them equally; the minimum and median over rounds are printed.

use std::hint::black_box;
use std::time::Instant;

use bench_common::Dataset;
use dataframes::kernels::{self, CmpOp, par};
use dataframes::memory::Buffer;

fn main() {
    let rows: usize = std::env::args().nth(1).map(|s| s.replace('_', "").parse().unwrap()).unwrap_or(1_000_000);
    let rounds: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(40);
    let ds = Dataset::generate(rows);
    let a = Buffer::from_slice(&ds.a);
    let threads = par::threads();
    println!("rows={rows} threads={threads} cache={:?}", kernels::cache::cache_info());

    for (pct, thr) in [(10, 100i64), (50, 500), (90, 900)] {
        type Variant<'a> = (&'a str, Box<dyn Fn() -> usize + 'a>);
        let variants: Vec<Variant> = vec![
            ("2phase par   plain ", Box::new(|| kernels::filter_by_cmp_scalar_with(&a, CmpOp::Lt, thr, threads, Some(false)).len())),
            ("2phase par   stream", Box::new(|| kernels::filter_by_cmp_scalar_with(&a, CmpOp::Lt, thr, threads, Some(true)).len())),
            ("2phase par/2 stream", Box::new(|| kernels::filter_by_cmp_scalar_with(&a, CmpOp::Lt, thr, threads / 2, Some(true)).len())),
            ("fused serial plain ", Box::new(|| kernels::filter_by_cmp_scalar_with(&a, CmpOp::Lt, thr, 1, Some(false)).len())),
            ("fused serial stream", Box::new(|| kernels::filter_by_cmp_scalar_with(&a, CmpOp::Lt, thr, 1, Some(true)).len())),
            ("fused par    plain ", Box::new(|| kernels::pipeline::filter_fused(&a, CmpOp::Lt, thr, threads, Some(false)).len())),
            ("fused par    stream", Box::new(|| kernels::pipeline::filter_fused(&a, CmpOp::Lt, thr, threads, Some(true)).len())),
            ("fused par/2  stream", Box::new(|| kernels::pipeline::filter_fused(&a, CmpOp::Lt, thr, threads / 2, Some(true)).len())),
            ("2phase serial plain", Box::new(|| {
                let (m, _) = kernels::cmp_scalar_mask_with(&a, CmpOp::Lt, thr, 1);
                kernels::filter_by_mask_with(&a, &m, 1, Some(false)).len()
            })),
            ("mask par + compact par stream", Box::new(|| {
                let (m, _) = kernels::cmp_scalar_mask_with(&a, CmpOp::Lt, thr, threads);
                kernels::filter_by_mask_with(&a, &m, threads, Some(true)).len()
            })),
        ];
        let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
        for round in 0..rounds + 2 {
            for (i, (_, f)) in variants.iter().enumerate() {
                let t = Instant::now();
                black_box(f());
                let dt = t.elapsed().as_nanos() as f64;
                if round >= 2 {
                    times[i].push(dt);
                }
            }
        }
        println!("-- filter i64 {pct}%");
        for (i, (name, _)) in variants.iter().enumerate() {
            let t = &mut times[i];
            t.sort_by(|x, y| x.total_cmp(y));
            println!("  {name:<30} min {:>9.1} µs   median {:>9.1} µs", t[0] / 1e3, t[t.len() / 2] / 1e3);
        }
    }
}
