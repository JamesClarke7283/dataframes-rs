//! Streaming-read experiment: does memory placement matter? (development tool)
use std::hint::black_box;
use std::time::Instant;

use dataframes::kernels::aggregate;
use dataframes::memory::Buffer;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn best<R>(mut f: impl FnMut() -> R) -> f64 {
    let mut b = f64::MAX;
    for _ in 0..10 {
        let t = Instant::now();
        black_box(f());
        b = b.min(t.elapsed().as_secs_f64() * 1e3);
    }
    b
}

fn main() {
    let n: usize = std::env::args().nth(1).map(|s| s.parse().unwrap()).unwrap_or(10_000_000);
    dataframes::kernels::par::set_min_task_bytes(usize::MAX);
    let v: Vec<f64> = (0..n).map(|i| i as f64 * 0.5).collect();
    let col_a = Buffer::from_slice(&v);
    let vec_b = v.clone();
    let col_c = Buffer::from_slice(&v);
    let vec_d = v.clone();
    println!("vec@{:p} buf@{:p}", v.as_ptr(), col_a.as_ptr());
    let regions: [(&str, &[f64]); 5] =
        [("vec  v", &v), ("col  a", &col_a), ("vec  b", &vec_b), ("col  c", &col_c), ("vec  d", &vec_d)];
    for pass in 0..3 {
        for (name, r) in regions.iter() {
            let mm = best(|| aggregate::min_max(black_box(*r)));
            let s = best(|| aggregate::sum(black_box(*r)));
            println!("pass {pass}  {name} @{:>14p} (mod 2M = {:>7}, mod 4K = {:>4})  min_max {mm:.3} ms   sum {s:.3} ms",
                r.as_ptr(), r.as_ptr() as usize % (2 << 20), r.as_ptr() as usize % 4096);
        }
    }
}
