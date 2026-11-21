**dataframes is the fastest library in 65 of 75 comparisons**, level with the fastest (within 3%) in 5 and slower in 5. Geometric-mean speed-up over the fastest competitor: **2.05x** (smallest 0.85x, largest 30.70x).

Machine: Intel(R) Core(TM) i5-8265U CPU @ 1.60GHz · 8 threads · Linux 7.3.3 · 2026-11-21 02:24 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0

Times are the best median over the rounds (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Arrow (arrow-rs) | vs fastest competitor |
|---|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **14.28 µs** | 74.71 µs | 13.71 µs | **0.96x** slower (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **18.43 µs** | 81.64 µs | 17.40 µs | **0.94x** slower (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **40.07 µs** | 77.59 µs | 34.18 µs | **0.85x** slower (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **37.46 µs** | 105.94 µs | 58.26 µs | **1.56x** (Arrow (arrow-rs)) |
| filter i64 column, 90% selected | **40.42 µs** | 107.04 µs | 150.25 µs | **2.65x** (Polars (Rust crate)) |
| filter f64 column, 50% selected | **37.65 µs** | 135.37 µs | 62.69 µs | **1.67x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **19.73 µs** | 120.33 µs | 56.76 µs | **2.88x** (Arrow (arrow-rs)) |
| filter f32 column, 50% selected | **19.52 µs** | 136.80 µs | 62.54 µs | **3.20x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 10% | **91.95 µs** | 159.95 µs | 141.03 µs | **1.53x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **144.72 µs** | 282.54 µs | 618.39 µs | **1.95x** (Polars (Rust crate)) |
| filter frame with a string column, 50% | **163.26 µs** | 260.29 µs | 422.67 µs | **1.59x** (Polars (Rust crate)) |
| filter 6-column frame by expression, 50% | **125.51 µs** | 348.41 µs | 680.99 µs | **2.78x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **117.95 µs** | 357.16 µs | 332.26 µs | **2.82x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **49.02 µs** | 193.21 µs | 64.82 µs | **1.32x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **13.1 ns** | 613.9 ns | 345.2 ns | **26.28x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **12.9 ns** | 622.6 ns | 392.7 ns | **30.50x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **19.1 ns** | 68.4 ns | 29.9 ns | **1.57x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **1.13 µs** | 3.42 µs | 1.04 µs | **0.91x** slower (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **18.70 µs** | 26.86 µs | 22.64 µs | **1.21x** (Arrow (arrow-rs)) |
| sum of i64 column | **12.00 µs** | 12.02 µs | 13.81 µs | **1.00x** level (Polars (Rust crate)) |
| sum of f64 column | **11.87 µs** | 16.23 µs | 13.02 µs | **1.10x** (Arrow (arrow-rs)) |
| mean of f64 column | **11.87 µs** | 15.87 µs | 14.35 µs | **1.21x** (Arrow (arrow-rs)) |
| min and max of f64 column | **25.33 µs** | 34.15 µs | 71.14 µs | **1.35x** (Polars (Rust crate)) |
| min and max of i64 column | **32.29 µs** | 29.77 µs | 32.20 µs | **0.92x** slower (Polars (Rust crate)) |
| in-place update x = x*c + d | **19.17 µs** | 24.06 µs | 19.40 µs | **1.01x** level (Arrow (arrow-rs)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Arrow (arrow-rs) | vs fastest competitor |
|---|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **181.69 µs** | 958.92 µs | 317.90 µs | **1.75x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **226.73 µs** | 901.07 µs | 404.72 µs | **1.79x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **274.78 µs** | 1.28 ms | 826.02 µs | **3.01x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **499.82 µs** | 1.42 ms | 981.56 µs | **1.96x** (Arrow (arrow-rs)) |
| filter i64 column, 90% selected | **456.21 µs** | 1.55 ms | 1.61 ms | **3.40x** (Polars (Rust crate)) |
| filter f64 column, 50% selected | **481.08 µs** | 2.01 ms | 1.07 ms | **2.23x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **148.18 µs** | 1.24 ms | 597.63 µs | **4.03x** (Arrow (arrow-rs)) |
| filter f32 column, 50% selected | **160.89 µs** | 1.43 ms | 814.54 µs | **5.06x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 10% | **2.46 ms** | 3.38 ms | 3.99 ms | **1.38x** (Polars (Rust crate)) |
| filter 6-column frame by mask, 50% | **2.86 ms** | 3.98 ms | 6.86 ms | **1.39x** (Polars (Rust crate)) |
| filter frame with a string column, 50% | **3.08 ms** | 4.15 ms | 5.64 ms | **1.35x** (Polars (Rust crate)) |
| filter 6-column frame by expression, 50% | **2.27 ms** | 3.55 ms | 6.98 ms | **1.56x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **1.96 ms** | 3.25 ms | 4.79 ms | **1.65x** (Polars (Rust crate)) |
| sum of one column over filtered rows | **594.84 µs** | 1.83 ms | 1.21 ms | **2.03x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **13.1 ns** | 622.1 ns | 403.4 ns | **30.70x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **12.9 ns** | 753.3 ns | 343.2 ns | **26.64x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **19.3 ns** | 81.6 ns | 30.1 ns | **1.56x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **951.0 ns** | 3.46 µs | 1.04 µs | **1.09x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **15.34 µs** | 22.61 µs | 22.64 µs | **1.47x** (Polars (Rust crate)) |
| sum of i64 column | **250.04 µs** | 498.13 µs | 500.70 µs | **1.99x** (Polars (Rust crate)) |
| sum of f64 column | **264.94 µs** | 337.75 µs | 493.01 µs | **1.27x** (Polars (Rust crate)) |
| mean of f64 column | **256.33 µs** | 340.74 µs | 497.34 µs | **1.33x** (Polars (Rust crate)) |
| min and max of f64 column | **508.27 µs** | 783.81 µs | 1.10 ms | **1.54x** (Polars (Rust crate)) |
| min and max of i64 column | **360.93 µs** | 794.59 µs | 724.90 µs | **2.01x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **389.15 µs** | 391.93 µs | 389.19 µs | **1.00x** level (Arrow (arrow-rs)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Arrow (arrow-rs) | vs fastest competitor |
|---|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **3.88 ms** | 8.24 ms | 4.40 ms | **1.13x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **3.61 ms** | 10.60 ms | 6.17 ms | **1.71x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **3.74 ms** | 13.83 ms | 10.46 ms | **2.80x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **4.42 ms** | 15.53 ms | 11.75 ms | **2.66x** (Arrow (arrow-rs)) |
| filter i64 column, 90% selected | **5.56 ms** | 18.05 ms | 17.51 ms | **3.15x** (Arrow (arrow-rs)) |
| filter f64 column, 50% selected | **4.48 ms** | 18.63 ms | 12.24 ms | **2.73x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **2.46 ms** | 13.88 ms | 7.86 ms | **3.19x** (Arrow (arrow-rs)) |
| filter f32 column, 50% selected | **2.43 ms** | 18.17 ms | 8.34 ms | **3.43x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 10% | **27.46 ms** | 32.51 ms | 41.25 ms | **1.18x** (Polars (Rust crate)) |
| filter 6-column frame by mask, 50% | **28.55 ms** | 39.49 ms | 71.38 ms | **1.38x** (Polars (Rust crate)) |
| filter frame with a string column, 50% | **32.96 ms** | 41.97 ms | 59.00 ms | **1.27x** (Polars (Rust crate)) |
| filter 6-column frame by expression, 50% | **23.00 ms** | 33.69 ms | 74.02 ms | **1.46x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **21.05 ms** | 29.68 ms | 51.65 ms | **1.41x** (Polars (Rust crate)) |
| sum of one column over filtered rows | **6.69 ms** | 14.13 ms | 13.94 ms | **2.08x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **13.5 ns** | 750.1 ns | 344.3 ns | **25.58x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **12.9 ns** | 619.9 ns | 378.6 ns | **29.38x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **21.2 ns** | 67.2 ns | 33.3 ns | **1.57x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **1.05 µs** | 4.12 µs | 1.04 µs | **0.99x** level (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **17.59 µs** | 27.83 µs | 22.65 µs | **1.29x** (Arrow (arrow-rs)) |
| sum of i64 column | **4.15 ms** | 5.85 ms | 5.92 ms | **1.41x** (Polars (Rust crate)) |
| sum of f64 column | **4.29 ms** | 6.13 ms | 5.82 ms | **1.36x** (Arrow (arrow-rs)) |
| mean of f64 column | **4.60 ms** | 6.49 ms | 5.81 ms | **1.26x** (Arrow (arrow-rs)) |
| min and max of f64 column | **8.48 ms** | 8.71 ms | 11.57 ms | **1.03x** level (Polars (Rust crate)) |
| min and max of i64 column | **7.36 ms** | 8.86 ms | 8.72 ms | **1.18x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **6.28 ms** | 6.61 ms | 6.58 ms | **1.05x** (Arrow (arrow-rs)) |
