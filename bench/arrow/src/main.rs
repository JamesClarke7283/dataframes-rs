//! Benchmark driver: Apache Arrow (arrow-rs compute kernels).
//!
//! arrow-rs has no query engine of its own, so the expression cases are
//! written the way one composes its kernels by hand: comparison kernel ->
//! boolean kernel -> filter kernel -> aggregate kernel.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::hint::black_box;
use std::sync::Arc;

use arrow_arith::aggregate::{max, min, sum};
use arrow_arith::boolean::and;
use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow_ord::cmp::lt;
use arrow_schema::{DataType, Field, Schema};
use arrow_select::filter::{filter, filter_record_batch};
use bench_common::{Dataset, Opts, Suite, vocab};

fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
    let fields: Vec<Field> = columns.iter().map(|(n, a)| Field::new(*n, a.data_type().clone(), false)).collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns.into_iter().map(|(_, a)| a).collect()).expect("valid batch")
}

fn i64_col<'a>(b: &'a RecordBatch, name: &str) -> &'a Int64Array {
    b.column_by_name(name).unwrap().as_any().downcast_ref::<Int64Array>().unwrap()
}

/// Check value of a batch: its height plus the sum of its `e` column.
fn frame_check(b: &RecordBatch) -> f64 {
    b.num_rows() as f64 + sum(i64_col(b, "e")).unwrap_or(0) as f64
}

fn main() {
    let opts = Opts::from_args();
    let ds = Dataset::generate(opts.rows);
    let n = ds.rows;
    let mut suite = Suite::new("arrow-rs", &opts);

    let words = vocab();
    let id: ArrayRef = Arc::new(Int64Array::from(ds.id.clone()));
    let a = Int64Array::from(ds.a.clone());
    let b = Float64Array::from(ds.b.clone());
    let c = Int32Array::from(ds.c.clone());
    let d = Float32Array::from(ds.d.clone());
    let e = Int64Array::from(ds.e.clone());
    let s: ArrayRef = Arc::new(StringViewArray::from_iter_values(ds.s.iter().map(|&i| words[i as usize].as_str())));
    let (ar, br, cr, dr, er): (ArrayRef, ArrayRef, ArrayRef, ArrayRef, ArrayRef) =
        (Arc::new(a.clone()), Arc::new(b.clone()), Arc::new(c.clone()), Arc::new(d.clone()), Arc::new(e.clone()));
    let num = batch(vec![("id", id.clone()), ("a", ar.clone()), ("b", br.clone()), ("c", cr), ("d", dr), ("e", er)]);
    let with_str = batch(vec![("id", id), ("a", ar), ("b", br), ("s", s)]);
    assert_eq!(a.data_type(), &DataType::Int64);

    let lt_i64 = |arr: &Int64Array, v: i64| lt(arr, &Int64Array::new_scalar(v)).unwrap();
    let lt_f64 = |arr: &Float64Array, v: f64| lt(arr, &Float64Array::new_scalar(v)).unwrap();
    let trues = |m: &BooleanArray| m.true_count() as f64;

    // --- comparison only ---------------------------------------------------
    suite.bench("mask_i64", || lt_i64(&a, 500), |m| trues(&m));

    // --- single-column filter: predicate + compaction ---------------------
    for (pct, thr) in [(1, 10i64), (10, 100), (50, 500), (90, 900)] {
        suite.bench(
            &format!("filter_i64_{pct}pct"),
            || filter(&a, &lt_i64(&a, thr)).unwrap(),
            |out| {
                let out = out.as_any().downcast_ref::<Int64Array>().unwrap();
                out.len() as f64 + sum(out).unwrap_or(0) as f64
            },
        );
    }
    suite.bench("filter_f64_50pct", || filter(&b, &lt_f64(&b, 0.5)).unwrap(), |out| out.len() as f64);
    suite.bench(
        "filter_i32_50pct",
        || filter(&c, &lt(&c, &Int32Array::new_scalar(50)).unwrap()).unwrap(),
        |out| {
            let out = out.as_any().downcast_ref::<Int32Array>().unwrap();
            out.len() as f64 + sum(out).unwrap_or(0) as f64
        },
    );
    suite.bench("filter_f32_50pct", || filter(&d, &lt(&d, &Float32Array::new_scalar(0.5)).unwrap()).unwrap(), |out| out.len() as f64);

    // --- whole-frame filter ---------------------------------------------------
    for (pct, thr) in [(10, 100i64), (50, 500)] {
        suite.bench(
            &format!("frame_filter_{pct}pct"),
            || filter_record_batch(&num, &lt_i64(i64_col(&num, "a"), thr)).unwrap(),
            |out| frame_check(&out),
        );
    }
    suite.bench(
        "frame_filter_str_50pct",
        || filter_record_batch(&with_str, &lt_i64(i64_col(&with_str, "a"), 500)).unwrap(),
        |out| out.num_rows() as f64 + sum(i64_col(&out, "id")).unwrap_or(0) as f64,
    );
    suite.bench(
        "frame_filter_expr_50pct",
        || filter_record_batch(&num, &lt_i64(i64_col(&num, "a"), 500)).unwrap(),
        |out| frame_check(&out),
    );
    suite.bench(
        "frame_filter_and_25pct",
        || {
            let mask = and(&lt_i64(&a, 500), &lt_f64(&b, 0.5)).unwrap();
            filter_record_batch(&num, &mask).unwrap()
        },
        |out| frame_check(&out),
    );
    suite.bench(
        "filter_then_sum",
        || {
            let kept = filter(&b, &lt_i64(&a, 500)).unwrap();
            sum(kept.as_any().downcast_ref::<Float64Array>().unwrap()).unwrap_or(0.0)
        },
        |v| v,
    );

    // --- slicing -------------------------------------------------------------
    suite.bench("slice_frame", || num.slice(n / 4, n / 2), |out| frame_check(&out));
    suite.bench("head_frame", || num.slice(0, 10.min(n)), |out| out.num_rows() as f64);
    suite.bench("slice_series", || a.slice(n / 4, n / 2), |out| out.len() as f64 + out.value(0) as f64);

    // --- retrieval -----------------------------------------------------------
    suite.bench(
        "get_random_1k",
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
        "get_random_1k@checked",
        || {
            // The null-aware spelling, equivalent to `get(i) -> Option<_>` elsewhere.
            let mut acc = 0i64;
            for &i in &ds.lookups {
                let i = black_box(i as usize);
                acc = acc.wrapping_add(if e.is_valid(i) { e.value(i) } else { 0 });
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
                let column = num.column_by_name(black_box("e")).unwrap().as_any().downcast_ref::<Int64Array>().unwrap();
                acc = acc.wrapping_add(column.value(black_box(i as usize)));
            }
            acc
        },
        |acc| acc as f64,
    );
    suite.bench("sum_i64", || sum(&e).unwrap_or(0), |v| v as f64);
    suite.bench("sum_f64", || sum(&b).unwrap_or(0.0), |v| v);
    suite.bench("mean_f64", || sum(&b).unwrap_or(0.0) / (b.len() - b.null_count()) as f64, |v| v);
    suite.bench("min_max_f64", || min(&b).unwrap_or(0.0) + max(&b).unwrap_or(0.0), |v| v);
    suite.bench("min_max_i64", || min(&e).unwrap_or(0) + max(&e).unwrap_or(0), |v| v as f64);

    // --- mutation --------------------------------------------------------------
    {
        // `unary_mut` updates in place when the buffer is not shared.
        let mut own = Some(Float64Array::from(ds.b.clone()));
        suite.bench(
            "mutate_f64_inplace",
            || {
                let arr = own.take().expect("array present");
                own = Some(arr.unary_mut(|v| v * 1.0000001 + 0.5).expect("buffer is not shared"));
            },
            |()| 0.0,
        );
    }

    suite.finish();
}
