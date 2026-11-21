**dataframes is the fastest library in 73 of 75 comparisons**, level with the fastest (within 5%, or not in every round) in 2 and slower in none. Geometric-mean speed-up over the fastest competitor: **2.28x** (smallest 1.03x, largest 47.69x).

Machine: AMD EPYC 9V74 80-Core Processor · 4 threads · Linux 7.0.0-1019-azure · 2026-11-21 09:15 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the best median over the rounds (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **4.13 µs** | 48.82 µs | 15.10 µs | 7.66 µs | 35.65 µs | **1.86x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **7.80 µs** | 50.87 µs | 22.56 µs | 10.02 µs | 114.62 µs | **1.29x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **11.52 µs** | 55.59 µs | 30.70 µs | 15.67 µs | 219.14 µs | **1.36x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **11.99 µs** | 80.22 µs | 33.12 µs | 40.18 µs | 593.93 µs | **2.76x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **13.39 µs** | 80.91 µs | 36.34 µs | 56.47 µs | 1.00 ms | **2.71x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **11.93 µs** | 95.39 µs | 36.07 µs | 44.27 µs | 582.29 µs | **3.02x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **7.27 µs** | 78.84 µs | 19.00 µs | 38.28 µs | 580.70 µs | **2.61x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **7.34 µs** | 93.93 µs | 19.76 µs | 39.98 µs | 590.39 µs | **2.69x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **33.06 µs** | 111.46 µs | 144.06 µs | 57.37 µs | 288.82 µs | **1.74x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **52.41 µs** | 211.68 µs | 150.08 µs | 146.39 µs | 1.42 ms | **2.79x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **49.16 µs** | 150.95 µs | 196.89 µs | 123.91 µs | 1.62 ms | **2.52x** (Arrow (arrow-rs)) |
| filter 6-column frame by expression, 50% | **48.84 µs** | 196.86 µs | 183.30 µs | 146.60 µs | 1.42 ms | **3.00x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **54.21 µs** | 203.85 µs | 257.35 µs | 104.47 µs | 476.56 µs | **1.93x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **23.77 µs** | 189.66 µs | 196.61 µs | 46.21 µs | 318.14 µs | **1.94x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 484.6 ns | 1.16 µs | 181.8 ns | 9.27 µs | **44.97x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 469.1 ns | 1.05 µs | 181.7 ns | 25.31 µs | **47.69x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.4 ns | 645.2 ns | 14.4 ns | 6.52 µs | **1.28x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **558.8 ns** | 2.41 µs | 176.00 µs | 590.2 ns | 74.07 µs | **1.06x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.54 µs** | 18.84 µs | - | 18.15 µs | - | **1.72x** (Arrow (arrow-rs)) |
| sum of i64 column | **3.81 µs** | 6.79 µs | 8.17 µs | 6.78 µs | 17.32 µs | **1.78x** (Arrow (arrow-rs)) |
| sum of f64 column | **3.83 µs** | 8.44 µs | 8.52 µs | 6.78 µs | 41.73 µs | **1.77x** (Arrow (arrow-rs)) |
| mean of f64 column | **3.84 µs** | 8.42 µs | 7.88 µs | 6.78 µs | 70.61 µs | **1.77x** (Arrow (arrow-rs)) |
| min and max of f64 column | **7.69 µs** | 14.90 µs | 21.68 µs | 42.43 µs | 125.73 µs | **1.94x** (Polars (Rust crate)) |
| min and max of i64 column | **7.61 µs** | 14.16 µs | 21.44 µs | 13.55 µs | 34.01 µs | **1.78x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **3.87 µs** | 6.91 µs | 49.30 µs | 7.21 µs | 19.28 µs | **1.79x** (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **48.78 µs** | 495.87 µs | 104.25 µs | 92.48 µs | 142.16 µs | **1.90x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **70.61 µs** | 522.33 µs | 187.54 µs | 115.80 µs | 731.67 µs | **1.64x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **109.16 µs** | 590.72 µs | 239.88 µs | 219.79 µs | 2.54 ms | **2.01x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **123.67 µs** | 807.58 µs | 268.04 µs | 450.10 µs | 8.25 ms | **2.17x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **141.59 µs** | 805.62 µs | 299.30 µs | 1.00 ms | 4.48 ms | **2.11x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **123.56 µs** | 957.14 µs | 287.28 µs | 475.66 µs | 8.23 ms | **2.33x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **58.37 µs** | 797.85 µs | 140.61 µs | 418.33 µs | 8.11 ms | **2.41x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **58.24 µs** | 949.38 µs | 143.37 µs | 438.81 µs | 7.92 ms | **2.46x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **593.52 µs** | 1.16 ms | 903.82 µs | 1.61 ms | 2.67 ms | **1.52x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **901.91 µs** | 1.61 ms | 1.27 ms | 3.12 ms | 10.74 ms | **1.41x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **894.70 µs** | 1.52 ms | 1.37 ms | 2.82 ms | 12.27 ms | **1.53x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **889.51 µs** | 1.33 ms | 1.24 ms | 3.14 ms | 10.71 ms | **1.39x** (Polars (Python wheel)) |
| filter frame by `a < x & b < y`, 25% | **740.14 µs** | 1.40 ms | 1.10 ms | 2.64 ms | 3.80 ms | **1.49x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **105.65 µs** | 761.02 µs | 426.02 µs | 497.90 µs | 4.90 ms | **4.03x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 482.0 ns | 1.15 µs | 181.0 ns | 9.21 µs | **44.82x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **4.0 ns** | 469.7 ns | 1.06 µs | 181.5 ns | 25.27 µs | **45.85x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.2 ns | 639.5 ns | 14.3 ns | 6.70 µs | **1.28x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **575.7 ns** | 2.41 µs | 173.06 µs | 597.8 ns | 74.38 µs | **1.04x** level (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.54 µs** | 18.88 µs | - | 17.07 µs | - | **1.62x** (Arrow (arrow-rs)) |
| sum of i64 column | **45.83 µs** | 88.62 µs | 92.62 µs | 88.19 µs | 101.43 µs | **1.92x** (Arrow (arrow-rs)) |
| sum of f64 column | **45.99 µs** | 96.15 µs | 96.99 µs | 88.85 µs | 299.54 µs | **1.93x** (Arrow (arrow-rs)) |
| mean of f64 column | **45.91 µs** | 97.13 µs | 97.11 µs | 88.99 µs | 596.76 µs | **1.94x** (Arrow (arrow-rs)) |
| min and max of f64 column | **92.51 µs** | 173.52 µs | 207.40 µs | 432.38 µs | 1.03 ms | **1.88x** (Polars (Rust crate)) |
| min and max of i64 column | **91.66 µs** | 167.42 µs | 207.09 µs | 175.95 µs | 194.18 µs | **1.83x** (Polars (Rust crate)) |
| in-place update x = x*c + d | **47.33 µs** | 89.86 µs | 420.00 µs | 89.69 µs | 224.87 µs | **1.90x** (Arrow (arrow-rs)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.44 ms** | 5.06 ms | 2.05 ms | 2.08 ms | 2.39 ms | **1.43x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **1.59 ms** | 6.72 ms | 5.71 ms | 3.41 ms | 10.85 ms | **2.14x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **2.35 ms** | 8.09 ms | 4.71 ms | 5.07 ms | 29.81 ms | **2.00x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **2.67 ms** | 8.76 ms | 5.18 ms | 6.09 ms | 60.01 ms | **1.94x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **3.43 ms** | 9.08 ms | 6.05 ms | 11.31 ms | 39.69 ms | **1.77x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **2.63 ms** | 10.34 ms | 5.27 ms | 6.09 ms | 59.91 ms | **2.00x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **926.13 µs** | 7.98 ms | 2.63 ms | 4.67 ms | 60.33 ms | **2.84x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **930.49 µs** | 9.47 ms | 2.67 ms | 4.73 ms | 59.52 ms | **2.87x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **12.61 ms** | 16.25 ms | 13.91 ms | 25.25 ms | 40.00 ms | **1.10x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **13.84 ms** | 19.71 ms | 17.41 ms | 33.25 ms | 66.97 ms | **1.26x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **13.95 ms** | 20.20 ms | 17.88 ms | 30.31 ms | 104.18 ms | **1.28x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **12.37 ms** | 15.81 ms | 15.75 ms | 33.15 ms | 67.65 ms | **1.27x** (Polars (Python wheel)) |
| filter frame by `a < x & b < y`, 25% | **11.81 ms** | 15.17 ms | 13.84 ms | 29.42 ms | 50.49 ms | **1.17x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **3.59 ms** | 6.74 ms | 4.22 ms | 7.35 ms | 31.94 ms | **1.17x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 480.2 ns | 1.13 µs | 181.7 ns | 9.14 µs | **44.98x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.9 ns** | 468.3 ns | 1.03 µs | 181.1 ns | 25.31 µs | **46.46x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.3 ns | 635.0 ns | 14.3 ns | 6.75 µs | **1.28x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **580.8 ns** | 2.41 µs | 176.87 µs | 599.3 ns | 74.82 µs | **1.03x** level (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.54 µs** | 18.87 µs | - | 18.15 µs | - | **1.72x** (Arrow (arrow-rs)) |
| sum of i64 column | **1.39 ms** | 2.13 ms | 1.89 ms | 2.12 ms | 2.07 ms | **1.36x** (Polars (Python wheel)) |
| sum of f64 column | **1.40 ms** | 1.93 ms | 1.94 ms | 2.13 ms | 5.52 ms | **1.38x** (Polars (Rust crate)) |
| mean of f64 column | **1.39 ms** | 1.93 ms | 1.94 ms | 2.13 ms | 8.33 ms | **1.38x** (Polars (Rust crate)) |
| min and max of f64 column | **2.79 ms** | 3.66 ms | 3.92 ms | 4.55 ms | 16.26 ms | **1.31x** (Polars (Rust crate)) |
| min and max of i64 column | **2.78 ms** | 3.28 ms | 3.84 ms | 4.24 ms | 3.54 ms | **1.18x** (Polars (Rust crate)) |
| in-place update x = x*c + d | **1.88 ms** | 2.19 ms | 8.56 ms | 2.21 ms | 4.81 ms | **1.17x** (Polars (Rust crate)) |
