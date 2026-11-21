//! Writes the benchmark dataset to disk so that the Python driver can load
//! exactly the columns the Rust drivers generate.
//!
//!     datagen --rows N --out DIR
//!
//! Every column is a raw little-endian array (`<name>.<dtype>`); `vocab.txt`
//! holds one word per line and `s.u32` indexes into it.

use std::io::Write;
use std::path::Path;

use bench_common::{Dataset, vocab};

fn write_raw<T: Copy>(dir: &Path, name: &str, values: &[T]) {
    // SAFETY: the element types written here are plain numbers without padding.
    let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values)) };
    std::fs::write(dir.join(name), bytes).unwrap_or_else(|e| panic!("cannot write {name}: {e}"));
}

fn main() {
    let mut rows = 1_000_000usize;
    let mut out = String::from("bench/results/data");
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--rows" => rows = args.next().expect("--rows needs a value").replace('_', "").parse().expect("integer"),
            "--out" => out = args.next().expect("--out needs a value"),
            other => panic!("unknown argument {other:?}"),
        }
    }
    assert!(cfg!(target_endian = "little"), "the dump format is little-endian");
    let dir = Path::new(&out);
    std::fs::create_dir_all(dir).expect("cannot create output directory");
    let ds = Dataset::generate(rows);
    write_raw(dir, "id.i64", &ds.id);
    write_raw(dir, "a.i64", &ds.a);
    write_raw(dir, "b.f64", &ds.b);
    write_raw(dir, "c.i32", &ds.c);
    write_raw(dir, "d.f32", &ds.d);
    write_raw(dir, "e.i64", &ds.e);
    write_raw(dir, "k.i32", &ds.k);
    write_raw(dir, "s.u32", &ds.s);
    write_raw(dir, "lookups.u32", &ds.lookups);
    let mut f = std::fs::File::create(dir.join("vocab.txt")).expect("cannot write vocab");
    for w in vocab() {
        writeln!(f, "{w}").expect("cannot write vocab");
    }
    std::fs::write(dir.join("rows.txt"), rows.to_string()).expect("cannot write rows");
    eprintln!("wrote {rows} rows to {}", dir.display());
}
