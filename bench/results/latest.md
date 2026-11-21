**dataframes is the fastest library in 55 of 75 comparisons**, level with the fastest (within 3%) in 6 and slower in 14. Geometric-mean speed-up over the fastest competitor: **1.87x** (smallest 0.70x, largest 56.16x).

Machine: AMD EPYC 7763 64-Core Processor · 4 threads · Linux 7.0.0-1020-azure · 2026-11-21 02:56 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the best median over the rounds (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **12.47 µs** | 55.56 µs | 20.21 µs | 10.35 µs | 52.13 µs | **0.83x** slower (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **15.99 µs** | 59.09 µs | 25.73 µs | 13.41 µs | 191.52 µs | **0.84x** slower (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **32.80 µs** | 67.66 µs | 35.73 µs | 22.86 µs | 403.82 µs | **0.70x** slower (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **36.18 µs** | 90.01 µs | 61.64 µs | 67.06 µs | 724.97 µs | **1.70x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **36.80 µs** | 90.75 µs | 61.38 µs | 66.14 µs | 1.35 ms | **1.67x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **36.46 µs** | 95.32 µs | 64.75 µs | 64.69 µs | 724.45 µs | **1.77x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **17.46 µs** | 91.24 µs | 55.47 µs | 62.82 µs | 726.22 µs | **3.18x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **17.50 µs** | 100.32 µs | 56.64 µs | 71.69 µs | 724.49 µs | **3.24x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **88.31 µs** | 134.65 µs | 185.95 µs | 75.94 µs | 422.07 µs | **0.86x** slower (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **98.54 µs** | 241.03 µs | 316.39 µs | 188.05 µs | 1.86 ms | **1.91x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **141.45 µs** | 173.05 µs | 234.95 µs | 165.02 µs | 2.06 ms | **1.17x** (Arrow (arrow-rs)) |
| filter 6-column frame by expression, 50% | **120.02 µs** | 242.51 µs | 358.13 µs | 188.03 µs | 1.85 ms | **1.57x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **128.39 µs** | 248.19 µs | 362.42 µs | 124.96 µs | 673.33 µs | **0.97x** level (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **47.22 µs** | 228.20 µs | 282.28 µs | 66.53 µs | 450.61 µs | **1.41x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **3.9 ns** | 620.8 ns | 1.49 µs | 215.7 ns | 13.01 µs | **55.38x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 602.5 ns | 1.38 µs | 205.2 ns | 39.67 µs | **54.51x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **19.2 ns** | 74.5 ns | 918.0 ns | 20.4 ns | 9.10 µs | **1.06x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **848.5 ns** | 3.40 µs | 232.78 µs | 759.5 ns | 91.58 µs | **0.90x** slower (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.26 µs** | 23.19 µs | - | 23.19 µs | - | **1.75x** (Polars (Rust crate)) |
| sum of i64 column | **9.43 µs** | 9.41 µs | 10.54 µs | 9.39 µs | 22.97 µs | **1.00x** level (Arrow (arrow-rs)) |
| sum of f64 column | **9.68 µs** | 11.51 µs | 11.25 µs | 11.66 µs | 52.36 µs | **1.16x** (Polars (Python wheel)) |
| mean of f64 column | **9.70 µs** | 11.52 µs | 11.00 µs | 11.66 µs | 86.85 µs | **1.13x** (Polars (Python wheel)) |
| min and max of f64 column | **19.06 µs** | 21.10 µs | 23.88 µs | 58.37 µs | 152.44 µs | **1.11x** (Polars (Rust crate)) |
| min and max of i64 column | **18.98 µs** | 19.70 µs | 32.74 µs | 18.89 µs | 48.63 µs | **1.00x** level (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **10.37 µs** | 10.41 µs | 61.71 µs | 10.77 µs | 28.08 µs | **1.00x** level (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **89.62 µs** | 561.24 µs | 119.62 µs | 100.92 µs | 162.45 µs | **1.13x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **170.53 µs** | 596.72 µs | 156.55 µs | 148.95 µs | 880.46 µs | **0.87x** slower (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **231.37 µs** | 722.55 µs | 282.78 µs | 296.29 µs | 2.91 ms | **1.22x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **225.13 µs** | 901.34 µs | 509.32 µs | 672.27 µs | 9.81 ms | **2.26x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **227.12 µs** | 903.87 µs | 511.73 µs | 1.21 ms | 5.48 ms | **2.25x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **227.05 µs** | 951.01 µs | 557.96 µs | 651.20 µs | 9.84 ms | **2.46x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **130.21 µs** | 916.03 µs | 454.03 µs | 633.12 µs | 9.76 ms | **3.49x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **134.15 µs** | 1.01 ms | 463.46 µs | 720.43 µs | 9.63 ms | **3.45x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **852.68 µs** | 1.31 ms | 1.17 ms | 1.57 ms | 2.79 ms | **1.37x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **1.23 ms** | 1.95 ms | 1.85 ms | 2.89 ms | 13.50 ms | **1.51x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **1.42 ms** | 1.81 ms | 1.73 ms | 2.71 ms | 14.48 ms | **1.22x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **1.34 ms** | 1.54 ms | 1.72 ms | 2.91 ms | 12.52 ms | **1.15x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **1.30 ms** | 1.64 ms | 1.54 ms | 2.44 ms | 4.29 ms | **1.19x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **240.26 µs** | 907.31 µs | 628.56 µs | 678.35 µs | 6.27 ms | **2.62x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **3.9 ns** | 613.1 ns | 1.42 µs | 214.8 ns | 12.98 µs | **55.18x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 596.8 ns | 1.34 µs | 212.5 ns | 39.11 µs | **56.16x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **18.9 ns** | 75.7 ns | 920.7 ns | 19.3 ns | 9.15 µs | **1.02x** level (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **852.8 ns** | 3.40 µs | 231.89 µs | 775.8 ns | 92.07 µs | **0.91x** slower (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.26 µs** | 23.21 µs | - | 23.16 µs | - | **1.75x** (Arrow (arrow-rs)) |
| sum of i64 column | **114.05 µs** | 96.58 µs | 99.44 µs | 96.87 µs | 113.79 µs | **0.85x** slower (Polars (Rust crate)) |
| sum of f64 column | **114.42 µs** | 115.52 µs | 105.62 µs | 116.42 µs | 361.72 µs | **0.92x** slower (Polars (Python wheel)) |
| mean of f64 column | **114.40 µs** | 115.26 µs | 107.64 µs | 116.46 µs | 703.11 µs | **0.94x** slower (Polars (Python wheel)) |
| min and max of f64 column | **229.60 µs** | 210.24 µs | 232.22 µs | 591.26 µs | 1.19 ms | **0.92x** slower (Polars (Rust crate)) |
| min and max of i64 column | **136.48 µs** | 197.11 µs | 314.68 µs | 194.69 µs | 228.40 µs | **1.43x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **84.63 µs** | 103.94 µs | 527.47 µs | 104.18 µs | 287.85 µs | **1.23x** (Polars (Rust crate)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.70 ms** | 5.77 ms | 2.93 ms | 2.96 ms | 3.18 ms | **1.72x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **2.57 ms** | 7.31 ms | 4.40 ms | 4.29 ms | 12.18 ms | **1.67x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **2.92 ms** | 8.75 ms | 5.93 ms | 6.20 ms | 35.15 ms | **2.03x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **5.07 ms** | 10.22 ms | 7.49 ms | 8.88 ms | 71.12 ms | **1.47x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **5.31 ms** | 10.65 ms | 8.39 ms | 14.09 ms | 49.20 ms | **1.58x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **5.24 ms** | 10.78 ms | 8.18 ms | 9.02 ms | 70.94 ms | **1.56x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **1.51 ms** | 9.33 ms | 5.28 ms | 7.02 ms | 71.13 ms | **3.49x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **1.56 ms** | 10.26 ms | 5.53 ms | 7.49 ms | 69.48 ms | **3.55x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **14.10 ms** | 17.24 ms | 14.99 ms | 19.69 ms | 41.88 ms | **1.06x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **17.33 ms** | 21.83 ms | 20.29 ms | 32.29 ms | 80.60 ms | **1.17x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **17.41 ms** | 22.93 ms | 21.58 ms | 30.36 ms | 120.00 ms | **1.24x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **19.90 ms** | 17.64 ms | 18.26 ms | 32.73 ms | 81.87 ms | **0.89x** slower (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **20.66 ms** | 16.51 ms | 16.13 ms | 27.79 ms | 59.20 ms | **0.78x** slower (Polars (Python wheel)) |
| sum of one column over filtered rows | **3.98 ms** | 7.12 ms | 4.56 ms | 9.64 ms | 41.81 ms | **1.15x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **3.9 ns** | 612.5 ns | 1.42 µs | 213.6 ns | 13.01 µs | **54.93x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 595.3 ns | 1.31 µs | 205.0 ns | 39.38 µs | **54.46x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **19.2 ns** | 74.2 ns | 914.8 ns | 19.2 ns | 9.04 µs | **1.00x** level (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **854.0 ns** | 3.40 µs | 233.61 µs | 773.8 ns | 91.38 µs | **0.91x** slower (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.26 µs** | 23.22 µs | - | 23.77 µs | - | **1.75x** (Polars (Rust crate)) |
| sum of i64 column | **1.50 ms** | 2.44 ms | 2.53 ms | 2.42 ms | 3.05 ms | **1.61x** (Arrow (arrow-rs)) |
| sum of f64 column | **1.48 ms** | 3.05 ms | 2.66 ms | 2.44 ms | 7.40 ms | **1.64x** (Arrow (arrow-rs)) |
| mean of f64 column | **1.50 ms** | 3.07 ms | 2.60 ms | 2.43 ms | 10.63 ms | **1.62x** (Arrow (arrow-rs)) |
| min and max of f64 column | **3.07 ms** | 6.31 ms | 6.33 ms | 7.55 ms | 20.98 ms | **2.05x** (Polars (Rust crate)) |
| min and max of i64 column | **3.24 ms** | 5.68 ms | 5.86 ms | 5.58 ms | 5.82 ms | **1.72x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **2.40 ms** | 2.88 ms | 13.31 ms | 2.85 ms | 6.15 ms | **1.19x** (Arrow (arrow-rs)) |
