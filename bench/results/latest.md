**dataframes is the fastest library in all 75 comparisons**: by more than 5% in 73, by less in 2. Geometric-mean speed-up over the fastest competitor: **2.29x** (smallest 1.04x, largest 49.26x).

Machine: AMD EPYC 9V74 80-Core Processor · 4 threads · Linux 7.0.0-1020-azure · 2026-11-21 12:05 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the mean over the rounds of each round's median (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **4.16 µs** | 48.82 µs | 15.37 µs | 7.59 µs | 35.96 µs | **1.83x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **7.49 µs** | 50.93 µs | 22.63 µs | 9.89 µs | 114.16 µs | **1.32x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **11.48 µs** | 55.79 µs | 30.91 µs | 15.67 µs | 220.75 µs | **1.36x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **11.91 µs** | 80.46 µs | 34.60 µs | 40.24 µs | 590.45 µs | **2.91x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **13.38 µs** | 80.97 µs | 37.22 µs | 56.32 µs | 1.04 ms | **2.78x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **11.87 µs** | 96.70 µs | 37.01 µs | 44.25 µs | 580.74 µs | **3.12x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **7.21 µs** | 79.10 µs | 19.72 µs | 38.24 µs | 580.29 µs | **2.73x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **7.29 µs** | 94.69 µs | 20.61 µs | 39.93 µs | 586.45 µs | **2.83x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **33.10 µs** | 118.04 µs | 143.68 µs | 57.44 µs | 293.04 µs | **1.74x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **52.04 µs** | 213.19 µs | 152.54 µs | 146.73 µs | 1.46 ms | **2.82x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **49.33 µs** | 151.23 µs | 196.57 µs | 124.05 µs | 1.66 ms | **2.51x** (Arrow (arrow-rs)) |
| filter 6-column frame by expression, 50% | **49.45 µs** | 195.87 µs | 184.58 µs | 146.58 µs | 1.47 ms | **2.96x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **54.62 µs** | 201.92 µs | 254.47 µs | 104.90 µs | 476.36 µs | **1.92x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **23.94 µs** | 192.74 µs | 195.35 µs | 46.21 µs | 320.29 µs | **1.93x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 487.9 ns | 1.15 µs | 199.1 ns | 9.31 µs | **49.26x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.9 ns** | 484.6 ns | 1.11 µs | 182.0 ns | 26.30 µs | **46.92x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 61.2 ns | 649.4 ns | 14.4 ns | 6.85 µs | **1.28x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **566.1 ns** | 2.42 µs | 169.20 µs | 595.1 ns | 76.93 µs | **1.05x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.55 µs** | 19.41 µs | - | 18.15 µs | - | **1.72x** (Arrow (arrow-rs)) |
| sum of i64 column | **3.81 µs** | 6.79 µs | 8.23 µs | 6.78 µs | 17.42 µs | **1.78x** (Arrow (arrow-rs)) |
| sum of f64 column | **3.83 µs** | 8.42 µs | 8.35 µs | 6.78 µs | 42.04 µs | **1.77x** (Arrow (arrow-rs)) |
| mean of f64 column | **3.83 µs** | 8.43 µs | 9.10 µs | 6.78 µs | 72.00 µs | **1.77x** (Arrow (arrow-rs)) |
| min and max of f64 column | **7.69 µs** | 14.90 µs | 22.45 µs | 42.50 µs | 128.31 µs | **1.94x** (Polars (Rust crate)) |
| min and max of i64 column | **7.61 µs** | 14.17 µs | 21.53 µs | 13.56 µs | 35.15 µs | **1.78x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **3.89 µs** | 6.91 µs | 49.24 µs | 7.23 µs | 22.26 µs | **1.78x** (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **48.95 µs** | 495.06 µs | 104.67 µs | 92.94 µs | 142.16 µs | **1.90x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **70.99 µs** | 521.91 µs | 188.23 µs | 115.81 µs | 734.28 µs | **1.63x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **109.16 µs** | 597.48 µs | 240.22 µs | 220.63 µs | 2.56 ms | **2.02x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **123.94 µs** | 806.79 µs | 269.13 µs | 452.16 µs | 8.42 ms | **2.17x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **141.85 µs** | 806.47 µs | 301.64 µs | 1.01 ms | 4.92 ms | **2.13x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **123.54 µs** | 966.90 µs | 289.82 µs | 475.13 µs | 8.51 ms | **2.35x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **58.51 µs** | 799.02 µs | 141.42 µs | 417.88 µs | 8.49 ms | **2.42x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **58.60 µs** | 960.61 µs | 144.55 µs | 438.84 µs | 8.13 ms | **2.47x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **602.14 µs** | 1.17 ms | 970.72 µs | 1.64 ms | 2.76 ms | **1.61x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **991.17 µs** | 1.69 ms | 1.32 ms | 3.16 ms | 11.34 ms | **1.33x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **920.35 µs** | 1.55 ms | 1.39 ms | 2.85 ms | 12.66 ms | **1.51x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **932.55 µs** | 1.34 ms | 1.23 ms | 3.15 ms | 10.93 ms | **1.32x** (Polars (Python wheel)) |
| filter frame by `a < x & b < y`, 25% | **739.30 µs** | 1.40 ms | 1.11 ms | 2.67 ms | 3.81 ms | **1.50x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **105.61 µs** | 758.60 µs | 422.36 µs | 497.37 µs | 5.11 ms | **4.00x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 497.9 ns | 1.15 µs | 183.4 ns | 9.29 µs | **45.43x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **4.0 ns** | 480.2 ns | 1.06 µs | 182.9 ns | 25.68 µs | **46.27x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.8 ns | 660.5 ns | 14.6 ns | 6.85 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **576.6 ns** | 2.43 µs | 169.90 µs | 599.0 ns | 79.01 µs | **1.04x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.54 µs** | 19.08 µs | - | 18.24 µs | - | **1.73x** (Arrow (arrow-rs)) |
| sum of i64 column | **45.83 µs** | 89.35 µs | 92.67 µs | 88.04 µs | 101.82 µs | **1.92x** (Arrow (arrow-rs)) |
| sum of f64 column | **45.94 µs** | 96.90 µs | 97.36 µs | 88.81 µs | 300.55 µs | **1.93x** (Arrow (arrow-rs)) |
| mean of f64 column | **45.87 µs** | 96.73 µs | 97.16 µs | 88.67 µs | 586.41 µs | **1.93x** (Arrow (arrow-rs)) |
| min and max of f64 column | **92.55 µs** | 174.61 µs | 208.08 µs | 430.77 µs | 1.04 ms | **1.89x** (Polars (Rust crate)) |
| min and max of i64 column | **91.71 µs** | 168.46 µs | 207.79 µs | 176.63 µs | 195.81 µs | **1.84x** (Polars (Rust crate)) |
| in-place update x = x*c + d | **46.88 µs** | 89.59 µs | 422.48 µs | 89.74 µs | 225.50 µs | **1.91x** (Polars (Rust crate)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.49 ms** | 5.08 ms | 2.07 ms | 2.18 ms | 2.40 ms | **1.38x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **1.62 ms** | 6.70 ms | 5.71 ms | 3.43 ms | 10.86 ms | **2.12x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **2.36 ms** | 8.09 ms | 4.81 ms | 5.14 ms | 30.49 ms | **2.04x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **2.71 ms** | 8.80 ms | 5.24 ms | 6.17 ms | 60.37 ms | **1.93x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **3.50 ms** | 9.18 ms | 6.07 ms | 11.35 ms | 40.54 ms | **1.73x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **2.69 ms** | 10.30 ms | 5.50 ms | 6.13 ms | 59.99 ms | **2.04x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **949.54 µs** | 8.00 ms | 2.66 ms | 4.67 ms | 60.45 ms | **2.80x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **956.35 µs** | 9.56 ms | 2.67 ms | 4.74 ms | 59.68 ms | **2.79x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **12.91 ms** | 16.50 ms | 14.10 ms | 25.32 ms | 41.41 ms | **1.09x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **14.07 ms** | 20.42 ms | 17.60 ms | 33.42 ms | 66.58 ms | **1.25x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **14.38 ms** | 20.42 ms | 18.08 ms | 30.41 ms | 105.34 ms | **1.26x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **12.65 ms** | 16.17 ms | 16.29 ms | 33.48 ms | 67.93 ms | **1.28x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **12.13 ms** | 15.53 ms | 14.13 ms | 29.77 ms | 51.67 ms | **1.16x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **3.68 ms** | 6.79 ms | 4.39 ms | 7.46 ms | 32.84 ms | **1.19x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 491.1 ns | 1.17 µs | 184.3 ns | 9.69 µs | **46.16x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **4.0 ns** | 476.8 ns | 1.03 µs | 182.0 ns | 26.03 µs | **45.86x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.8 ns | 647.9 ns | 14.9 ns | 6.87 µs | **1.32x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **583.3 ns** | 2.42 µs | 172.53 µs | 603.8 ns | 77.40 µs | **1.04x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.60 µs** | 19.29 µs | - | 18.25 µs | - | **1.72x** (Arrow (arrow-rs)) |
| sum of i64 column | **1.45 ms** | 2.01 ms | 1.92 ms | 2.18 ms | 2.12 ms | **1.32x** (Polars (Python wheel)) |
| sum of f64 column | **1.45 ms** | 2.02 ms | 1.96 ms | 2.19 ms | 5.61 ms | **1.35x** (Polars (Python wheel)) |
| mean of f64 column | **1.46 ms** | 1.97 ms | 1.97 ms | 2.18 ms | 8.49 ms | **1.35x** (Polars (Rust crate)) |
| min and max of f64 column | **2.88 ms** | 3.71 ms | 3.95 ms | 4.58 ms | 16.50 ms | **1.29x** (Polars (Rust crate)) |
| min and max of i64 column | **2.87 ms** | 3.66 ms | 3.94 ms | 4.04 ms | 3.59 ms | **1.25x** (pandas) |
| in-place update x = x*c + d | **1.93 ms** | 2.32 ms | 8.46 ms | 2.30 ms | 4.94 ms | **1.19x** (Arrow (arrow-rs)) |
