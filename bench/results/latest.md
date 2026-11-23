**dataframes is the fastest library in all 75 comparisons**: by more than 5% in 74, by less in 1. Geometric-mean speed-up over the fastest competitor: **2.29x** (smallest 1.03x, largest 50.29x).

Machine: AMD EPYC 9V74 80-Core Processor · 4 threads · Linux 7.0.0-1020-azure · 2026-11-23 14:52 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the mean over the rounds of each round's median (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **4.16 µs** | 48.82 µs | 14.99 µs | 7.68 µs | 36.18 µs | **1.85x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **7.49 µs** | 50.93 µs | 22.58 µs | 10.06 µs | 113.91 µs | **1.34x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **11.49 µs** | 55.75 µs | 30.81 µs | 15.76 µs | 238.75 µs | **1.37x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **11.97 µs** | 80.24 µs | 33.35 µs | 40.11 µs | 596.92 µs | **2.79x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **13.40 µs** | 80.77 µs | 36.91 µs | 57.26 µs | 1.05 ms | **2.75x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **11.85 µs** | 95.79 µs | 36.38 µs | 44.37 µs | 589.80 µs | **3.07x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **7.20 µs** | 78.86 µs | 20.22 µs | 38.34 µs | 583.29 µs | **2.81x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **7.30 µs** | 94.69 µs | 20.49 µs | 39.99 µs | 593.38 µs | **2.81x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **33.05 µs** | 116.61 µs | 149.71 µs | 57.36 µs | 293.82 µs | **1.74x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **51.41 µs** | 211.25 µs | 149.74 µs | 146.78 µs | 1.45 ms | **2.85x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **49.02 µs** | 158.06 µs | 197.91 µs | 123.99 µs | 1.65 ms | **2.53x** (Arrow (arrow-rs)) |
| filter 6-column frame by expression, 50% | **49.55 µs** | 196.79 µs | 184.37 µs | 147.10 µs | 1.46 ms | **2.97x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **54.70 µs** | 202.42 µs | 255.87 µs | 105.49 µs | 479.36 µs | **1.93x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **23.78 µs** | 195.73 µs | 199.57 µs | 46.33 µs | 320.71 µs | **1.95x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 500.3 ns | 1.17 µs | 184.1 ns | 9.50 µs | **45.46x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.9 ns** | 490.3 ns | 1.06 µs | 185.6 ns | 26.32 µs | **47.20x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.5 ns | 675.1 ns | 14.4 ns | 6.69 µs | **1.29x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **561.4 ns** | 2.41 µs | 186.71 µs | 594.5 ns | 78.43 µs | **1.06x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.61 µs** | 19.07 µs | - | 17.97 µs | - | **1.69x** (Arrow (arrow-rs)) |
| sum of i64 column | **3.80 µs** | 6.79 µs | 8.02 µs | 6.78 µs | 17.53 µs | **1.79x** (Arrow (arrow-rs)) |
| sum of f64 column | **3.82 µs** | 8.43 µs | 8.65 µs | 6.78 µs | 41.66 µs | **1.77x** (Arrow (arrow-rs)) |
| mean of f64 column | **3.83 µs** | 8.45 µs | 8.67 µs | 6.78 µs | 72.21 µs | **1.77x** (Arrow (arrow-rs)) |
| min and max of f64 column | **7.74 µs** | 14.90 µs | 22.61 µs | 42.55 µs | 125.59 µs | **1.93x** (Polars (Rust crate)) |
| min and max of i64 column | **7.59 µs** | 14.16 µs | 21.67 µs | 13.56 µs | 34.24 µs | **1.79x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **3.88 µs** | 6.91 µs | 49.54 µs | 7.27 µs | 20.33 µs | **1.78x** (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **48.93 µs** | 495.45 µs | 104.43 µs | 93.01 µs | 141.50 µs | **1.90x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **70.85 µs** | 521.70 µs | 188.86 µs | 115.01 µs | 731.59 µs | **1.62x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **109.12 µs** | 591.59 µs | 239.88 µs | 221.24 µs | 2.56 ms | **2.03x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **123.78 µs** | 807.34 µs | 268.37 µs | 450.51 µs | 8.46 ms | **2.17x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **141.65 µs** | 807.18 µs | 302.59 µs | 1.00 ms | 4.68 ms | **2.14x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **123.57 µs** | 966.02 µs | 287.07 µs | 474.63 µs | 8.54 ms | **2.32x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **58.40 µs** | 798.03 µs | 140.79 µs | 418.09 µs | 8.41 ms | **2.41x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **58.39 µs** | 954.54 µs | 144.79 µs | 439.16 µs | 8.16 ms | **2.48x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **630.27 µs** | 1.20 ms | 974.57 µs | 1.72 ms | 2.79 ms | **1.55x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **957.69 µs** | 1.65 ms | 1.36 ms | 3.17 ms | 10.98 ms | **1.42x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **969.00 µs** | 1.59 ms | 1.47 ms | 2.84 ms | 12.54 ms | **1.52x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **966.80 µs** | 1.42 ms | 1.31 ms | 3.17 ms | 11.21 ms | **1.36x** (Polars (Python wheel)) |
| filter frame by `a < x & b < y`, 25% | **781.61 µs** | 1.44 ms | 1.19 ms | 2.67 ms | 3.90 ms | **1.52x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **105.68 µs** | 765.15 µs | 450.25 µs | 499.77 µs | 5.12 ms | **4.26x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.1 ns** | 492.5 ns | 1.15 µs | 183.2 ns | 9.35 µs | **45.21x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.9 ns** | 483.2 ns | 1.06 µs | 185.3 ns | 26.30 µs | **47.07x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.7 ns | 654.6 ns | 14.6 ns | 6.71 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **576.2 ns** | 2.42 µs | 174.33 µs | 605.7 ns | 75.89 µs | **1.05x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.60 µs** | 18.91 µs | - | 18.24 µs | - | **1.72x** (Arrow (arrow-rs)) |
| sum of i64 column | **45.92 µs** | 89.10 µs | 92.57 µs | 88.31 µs | 101.77 µs | **1.92x** (Arrow (arrow-rs)) |
| sum of f64 column | **45.93 µs** | 96.38 µs | 97.06 µs | 88.87 µs | 299.24 µs | **1.93x** (Arrow (arrow-rs)) |
| mean of f64 column | **45.95 µs** | 96.45 µs | 97.34 µs | 89.55 µs | 591.13 µs | **1.95x** (Arrow (arrow-rs)) |
| min and max of f64 column | **92.45 µs** | 174.02 µs | 207.94 µs | 431.03 µs | 1.05 ms | **1.88x** (Polars (Rust crate)) |
| min and max of i64 column | **91.70 µs** | 167.12 µs | 208.05 µs | 176.14 µs | 195.03 µs | **1.82x** (Polars (Rust crate)) |
| in-place update x = x*c + d | **47.20 µs** | 89.42 µs | 449.35 µs | 89.40 µs | 225.53 µs | **1.89x** (Arrow (arrow-rs)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.46 ms** | 5.11 ms | 2.08 ms | 2.13 ms | 2.41 ms | **1.42x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **1.62 ms** | 6.75 ms | 5.86 ms | 3.45 ms | 10.93 ms | **2.13x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **2.40 ms** | 8.20 ms | 4.76 ms | 5.12 ms | 30.36 ms | **1.98x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **2.67 ms** | 8.82 ms | 5.34 ms | 6.08 ms | 60.81 ms | **2.00x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **3.53 ms** | 9.12 ms | 6.29 ms | 11.36 ms | 40.57 ms | **1.78x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **2.99 ms** | 10.38 ms | 5.44 ms | 6.14 ms | 60.11 ms | **1.82x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **965.85 µs** | 8.02 ms | 2.69 ms | 4.66 ms | 60.72 ms | **2.79x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **972.28 µs** | 9.57 ms | 2.77 ms | 4.75 ms | 60.31 ms | **2.85x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **12.73 ms** | 16.90 ms | 14.03 ms | 25.27 ms | 40.77 ms | **1.10x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **13.97 ms** | 20.26 ms | 17.71 ms | 33.84 ms | 67.24 ms | **1.27x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **13.85 ms** | 20.42 ms | 18.14 ms | 30.89 ms | 105.91 ms | **1.31x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **12.50 ms** | 16.24 ms | 16.11 ms | 33.80 ms | 68.65 ms | **1.29x** (Polars (Python wheel)) |
| filter frame by `a < x & b < y`, 25% | **11.91 ms** | 15.36 ms | 14.02 ms | 30.13 ms | 52.01 ms | **1.18x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **3.60 ms** | 6.76 ms | 4.31 ms | 7.66 ms | 33.52 ms | **1.20x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 491.2 ns | 1.13 µs | 183.5 ns | 9.57 µs | **45.37x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.9 ns** | 480.1 ns | 1.06 µs | 196.7 ns | 25.97 µs | **50.29x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **11.2 ns** | 60.6 ns | 661.4 ns | 14.5 ns | 6.80 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **586.9 ns** | 2.42 µs | 174.15 µs | 607.0 ns | 76.75 µs | **1.03x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **10.55 µs** | 18.98 µs | - | 17.49 µs | - | **1.66x** (Arrow (arrow-rs)) |
| sum of i64 column | **1.37 ms** | 2.12 ms | 1.93 ms | 2.12 ms | 2.11 ms | **1.41x** (Polars (Python wheel)) |
| sum of f64 column | **1.42 ms** | 1.95 ms | 1.98 ms | 2.10 ms | 5.60 ms | **1.37x** (Polars (Rust crate)) |
| mean of f64 column | **1.52 ms** | 2.01 ms | 1.95 ms | 2.15 ms | 8.51 ms | **1.28x** (Polars (Python wheel)) |
| min and max of f64 column | **2.81 ms** | 3.67 ms | 3.94 ms | 4.57 ms | 16.32 ms | **1.30x** (Polars (Rust crate)) |
| min and max of i64 column | **2.85 ms** | 4.03 ms | 3.87 ms | 4.25 ms | 3.66 ms | **1.28x** (pandas) |
| in-place update x = x*c + d | **2.00 ms** | 2.25 ms | 9.24 ms | 2.22 ms | 4.80 ms | **1.11x** (Arrow (arrow-rs)) |
