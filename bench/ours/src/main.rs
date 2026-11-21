//! Benchmark driver: dataframes (this repository).
//!
//! The cases mirror `bench/polars/src/main.rs` one for one and are written
//! against the public API only, with the same call shapes a Polars user
//! would write.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;

use bench_common::{Dataset, Opts, Suite, vocab};
use dataframes::prelude::*;

fn build(ds: &Dataset) -> DataFrame {
    let words = vocab();
    let s: Vec<&str> = ds.s.iter().map(|&i| words[i as usize].as_str()).collect();
    DataFrame::new(
        ds.rows,
        vec![
            Int64Chunked::from_slice("id".into(), &ds.id).into_column(),
            Int64Chunked::from_slice("a".into(), &ds.a).into_column(),
            Float64Chunked::from_slice("b".into(), &ds.b).into_column(),
            Int32Chunked::from_slice("c".into(), &ds.c).into_column(),
            Float32Chunked::from_slice("d".into(), &ds.d).into_column(),
            Int64Chunked::from_slice("e".into(), &ds.e).into_column(),
            Int32Chunked::from_slice("k".into(), &ds.k).into_column(),
            StringChunked::from_slice("s".into(), &s).into_column(),
        ],
    )
    .expect("valid frame")
}

/// Check value of a frame: its height plus the sum of its `e` column.
fn frame_check(df: &DataFrame) -> f64 {
    df.height() as f64 + df.column("e").unwrap().i64().unwrap().sum().unwrap_or(0) as f64
}

fn main() -> PolarsResult<()> {
    let opts = Opts::from_args();
    let ds = Dataset::generate(opts.rows);
    let full = build(&ds);
    let n = ds.rows;
    let mut suite = Suite::new("dataframes", &opts);

    let num = full.select(["id", "a", "b", "c", "d", "e"])?;
    let with_str = full.select(["id", "a", "b", "s"])?;
    let a = full.column("a")?.i64()?.clone();
    let b = full.column("b")?.f64()?.clone();
    let c = full.column("c")?.i32()?.clone();
    let d = full.column("d")?.f32()?.clone();
    let e = full.column("e")?.i64()?.clone();

    // --- comparison only ---------------------------------------------------
    suite.bench("mask_i64", || a.lt(500i64), |m| m.sum().unwrap() as f64);

    // --- single-column filter: predicate + compaction ---------------------
    for (pct, thr) in [(1, 10i64), (10, 100), (50, 500), (90, 900)] {
        suite.bench(
            &format!("filter_i64_{pct}pct"),
            || a.filter(&a.lt(thr)).unwrap(),
            |out| out.len() as f64 + out.sum().unwrap_or(0) as f64,
        );
        suite.bench(
            &format!("filter_i64_{pct}pct@fused"),
            || a.filter_cmp(CmpOp::Lt, thr),
            |out| out.len() as f64 + out.sum().unwrap_or(0) as f64,
        );
    }
    suite.bench("filter_f64_50pct", || b.filter(&b.lt(0.5f64)).unwrap(), |out| out.len() as f64);
    suite.bench("filter_f64_50pct@fused", || b.filter_cmp(CmpOp::Lt, 0.5f64), |out| out.len() as f64);
    suite.bench("filter_i32_50pct", || c.filter(&c.lt(50i32)).unwrap(), |out| out.len() as f64 + out.sum().unwrap_or(0) as f64);
    suite.bench("filter_i32_50pct@fused", || c.filter_cmp(CmpOp::Lt, 50i32), |out| out.len() as f64 + out.sum().unwrap_or(0) as f64);
    suite.bench("filter_f32_50pct", || d.filter(&d.lt(0.5f32)).unwrap(), |out| out.len() as f64);
    suite.bench("filter_f32_50pct@fused", || d.filter_cmp(CmpOp::Lt, 0.5f32), |out| out.len() as f64);

    // --- whole-frame filter, eager API ------------------------------------
    for (pct, thr) in [(10, 100i64), (50, 500)] {
        suite.bench(
            &format!("frame_filter_{pct}pct"),
            || {
                let mask = num.column("a").unwrap().i64().unwrap().lt(thr);
                num.filter(&mask).unwrap()
            },
            |out| frame_check(&out),
        );
    }
    suite.bench(
        "frame_filter_str_50pct",
        || {
            let mask = with_str.column("a").unwrap().i64().unwrap().lt(500i64);
            with_str.filter(&mask).unwrap()
        },
        |out| out.height() as f64 + out.column("id").unwrap().i64().unwrap().sum().unwrap_or(0) as f64,
    );

    // --- whole-frame filter, expression API --------------------------------
    suite.bench(
        "frame_filter_expr_50pct",
        || num.clone().lazy().filter(col("a").lt(lit(500i64))).collect().unwrap(),
        |out| frame_check(&out),
    );
    suite.bench(
        "frame_filter_and_25pct",
        || num.clone().lazy().filter(col("a").lt(lit(500i64)).and(col("b").lt(lit(0.5f64)))).collect().unwrap(),
        |out| frame_check(&out),
    );
    suite.bench(
        "filter_then_sum",
        || num.clone().lazy().filter(col("a").lt(lit(500i64))).select([col("b").sum()]).collect().unwrap(),
        |out| out.column("b").unwrap().f64().unwrap().get(0).unwrap(),
    );

    // --- slicing -------------------------------------------------------------
    suite.bench("slice_frame", || num.slice((n / 4) as i64, n / 2), |out| frame_check(&out));
    suite.bench("head_frame", || num.head(Some(10)), |out| out.height() as f64);
    suite.bench("slice_series", || a.slice((n / 4) as i64, n / 2), |out| out.len() as f64 + out.get(0).unwrap_or(0) as f64);

    // --- retrieval -----------------------------------------------------------
    suite.bench(
        "get_random_1k",
        || {
            let mut acc = 0i64;
            for &i in &ds.lookups {
                acc = acc.wrapping_add(e.get(black_box(i as usize)).unwrap_or(0));
            }
            acc
        },
        |acc| acc as f64,
    );
    suite.bench(
        "get_random_1k@value",
        || {
            let mut acc = 0i64;
            for &i in &ds.lookups {
                acc = acc.wrapping_add(e.value(black_box(i as usize)));
            }
            acc
        },
        |acc| acc as f64,
    );
    suite.bench(
        "cell_random_1k",
        || {
            let mut acc = 0i64;
            for &i in &ds.lookups {
                let column = num.column(black_box("e")).unwrap().i64().unwrap();
                acc = acc.wrapping_add(column.get(black_box(i as usize)).unwrap_or(0));
            }
            acc
        },
        |acc| acc as f64,
    );
    suite.bench("sum_i64", || e.sum().unwrap_or(0), |v| v as f64);
    suite.bench("sum_f64", || b.sum().unwrap_or(0.0), |v| v);
    suite.bench("mean_f64", || b.mean().unwrap_or(0.0), |v| v);
    suite.bench("min_max_f64", || b.min().unwrap_or(0.0) + b.max().unwrap_or(0.0), |v| v);
    suite.bench("min_max_i64", || e.min().unwrap_or(0) + e.max().unwrap_or(0), |v| v as f64);

    // --- mutation --------------------------------------------------------------
    {
        // A private copy: `apply_mut` works in place when the buffer is not shared.
        let mut own = Float64Chunked::from_slice("b".into(), &ds.b);
        suite.bench("mutate_f64_inplace", || own.apply_mut(|v| v * 1.0000001 + 0.5), |()| 0.0);
    }

    suite.finish();
    Ok(())
}
