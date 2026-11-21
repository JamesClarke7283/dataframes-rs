**dataframes is the fastest library in 75 of 75 comparisons** and slower in none. Geometric-mean speed-up over the fastest competitor: **2.62x** (smallest 1.15x, largest 58.59x).

Machine: AMD EPYC 7763 64-Core Processor · 4 threads · Linux 7.0.0-1020-azure · 2026-11-21 04:53 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the best median over the rounds (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **7.47 µs** | 55.07 µs | 20.30 µs | 10.26 µs | 52.51 µs | **1.37x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **11.94 µs** | 58.61 µs | 26.19 µs | 15.38 µs | 186.30 µs | **1.29x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **19.17 µs** | 67.16 µs | 40.76 µs | 22.73 µs | 401.45 µs | **1.19x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **22.15 µs** | 89.27 µs | 63.08 µs | 67.50 µs | 732.25 µs | **2.85x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **23.41 µs** | 90.02 µs | 61.69 µs | 67.02 µs | 1.36 ms | **2.63x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **22.01 µs** | 103.18 µs | 66.44 µs | 64.93 µs | 732.19 µs | **2.95x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **13.40 µs** | 90.76 µs | 55.87 µs | 62.92 µs | 724.96 µs | **4.17x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **13.37 µs** | 95.71 µs | 56.65 µs | 71.83 µs | 721.02 µs | **4.24x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **43.18 µs** | 132.41 µs | 192.70 µs | 76.48 µs | 435.22 µs | **1.77x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **78.06 µs** | 237.00 µs | 317.21 µs | 189.01 µs | 1.88 ms | **2.42x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **76.38 µs** | 175.08 µs | 236.13 µs | 166.00 µs | 2.04 ms | **2.17x** (Arrow (arrow-rs)) |
| filter 6-column frame by expression, 50% | **77.90 µs** | 238.93 µs | 357.22 µs | 190.54 µs | 1.84 ms | **2.45x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **64.07 µs** | 250.37 µs | 364.60 µs | 126.37 µs | 676.51 µs | **1.97x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **18.23 µs** | 226.60 µs | 271.90 µs | 69.76 µs | 455.01 µs | **3.83x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 593.6 ns | 1.47 µs | 212.1 ns | 12.79 µs | **53.44x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.7 ns** | 597.5 ns | 1.34 µs | 219.6 ns | 39.04 µs | **58.59x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **12.7 ns** | 71.3 ns | 891.6 ns | 16.5 ns | 8.99 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **638.0 ns** | 3.02 µs | 231.51 µs | 763.4 ns | 90.72 µs | **1.20x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.13 µs** | 23.60 µs | - | 23.49 µs | - | **1.79x** (Arrow (arrow-rs)) |
| sum of i64 column | **5.16 µs** | 9.45 µs | 10.42 µs | 9.42 µs | 23.12 µs | **1.83x** (Arrow (arrow-rs)) |
| sum of f64 column | **5.27 µs** | 10.86 µs | 10.92 µs | 11.66 µs | 52.16 µs | **2.06x** (Polars (Rust crate)) |
| mean of f64 column | **5.09 µs** | 10.98 µs | 11.79 µs | 11.67 µs | 88.31 µs | **2.15x** (Polars (Rust crate)) |
| min and max of f64 column | **10.65 µs** | 20.66 µs | 23.74 µs | 58.32 µs | 151.15 µs | **1.94x** (Polars (Rust crate)) |
| min and max of i64 column | **10.91 µs** | 19.47 µs | 32.60 µs | 23.47 µs | 52.99 µs | **1.79x** (Polars (Rust crate)) |
| in-place update x = x*c + d | **6.20 µs** | 10.42 µs | 61.76 µs | 10.76 µs | 27.59 µs | **1.68x** (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **63.01 µs** | 555.08 µs | 118.26 µs | 103.04 µs | 163.75 µs | **1.64x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **90.36 µs** | 589.83 µs | 157.09 µs | 145.90 µs | 879.67 µs | **1.61x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **141.18 µs** | 716.38 µs | 284.17 µs | 296.26 µs | 2.92 ms | **2.01x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **191.10 µs** | 894.34 µs | 512.02 µs | 673.60 µs | 9.76 ms | **2.68x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **191.83 µs** | 895.58 µs | 511.18 µs | 1.21 ms | 5.32 ms | **2.66x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **190.97 µs** | 1.04 ms | 558.05 µs | 653.29 µs | 9.69 ms | **2.92x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **105.62 µs** | 911.21 µs | 455.10 µs | 634.08 µs | 9.55 ms | **4.31x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **105.67 µs** | 962.00 µs | 463.95 µs | 721.80 µs | 9.54 ms | **4.39x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **560.31 µs** | 1.26 ms | 1.11 ms | 1.47 ms | 2.63 ms | **1.97x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **947.76 µs** | 1.94 ms | 1.78 ms | 2.76 ms | 12.28 ms | **1.87x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **924.00 µs** | 1.69 ms | 1.63 ms | 2.51 ms | 14.69 ms | **1.76x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **931.43 µs** | 1.50 ms | 1.67 ms | 2.71 ms | 12.18 ms | **1.61x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **851.57 µs** | 1.61 ms | 1.48 ms | 2.34 ms | 4.10 ms | **1.74x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **124.61 µs** | 909.43 µs | 625.27 µs | 668.93 µs | 6.03 ms | **5.02x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 602.4 ns | 1.47 µs | 204.7 ns | 12.71 µs | **51.56x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.7 ns** | 585.3 ns | 1.36 µs | 206.6 ns | 39.40 µs | **55.25x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **12.7 ns** | 71.6 ns | 879.6 ns | 16.5 ns | 9.08 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **655.7 ns** | 3.02 µs | 237.86 µs | 770.6 ns | 89.82 µs | **1.18x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.12 µs** | 23.59 µs | - | 23.49 µs | - | **1.79x** (Arrow (arrow-rs)) |
| sum of i64 column | **50.73 µs** | 96.93 µs | 98.56 µs | 97.05 µs | 113.57 µs | **1.91x** (Polars (Rust crate)) |
| sum of f64 column | **51.44 µs** | 113.22 µs | 105.57 µs | 116.99 µs | 362.89 µs | **2.05x** (Polars (Python wheel)) |
| mean of f64 column | **51.39 µs** | 114.14 µs | 105.62 µs | 116.42 µs | 682.75 µs | **2.06x** (Polars (Python wheel)) |
| min and max of f64 column | **102.36 µs** | 207.16 µs | 229.13 µs | 592.73 µs | 1.19 ms | **2.02x** (Polars (Rust crate)) |
| min and max of i64 column | **102.05 µs** | 198.68 µs | 315.95 µs | 195.23 µs | 231.02 µs | **1.91x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **51.02 µs** | 103.78 µs | 470.12 µs | 104.79 µs | 289.04 µs | **2.03x** (Polars (Rust crate)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.34 ms** | 5.75 ms | 2.76 ms | 2.90 ms | 3.18 ms | **2.06x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **1.49 ms** | 7.17 ms | 4.19 ms | 4.21 ms | 11.47 ms | **2.81x** (Polars (Python wheel)) |
| filter i64 column, 10% selected | **2.34 ms** | 8.59 ms | 5.61 ms | 6.10 ms | 35.19 ms | **2.40x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **2.94 ms** | 10.03 ms | 7.29 ms | 8.90 ms | 70.62 ms | **2.48x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **3.71 ms** | 10.26 ms | 7.98 ms | 14.00 ms | 48.26 ms | **2.15x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **2.97 ms** | 11.30 ms | 8.32 ms | 8.77 ms | 70.22 ms | **2.81x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **1.47 ms** | 9.27 ms | 5.21 ms | 6.91 ms | 70.44 ms | **3.55x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **1.47 ms** | 9.82 ms | 5.44 ms | 7.45 ms | 69.03 ms | **3.69x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **11.15 ms** | 16.20 ms | 13.95 ms | 19.95 ms | 40.89 ms | **1.25x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **13.55 ms** | 21.63 ms | 18.89 ms | 31.68 ms | 79.84 ms | **1.39x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **14.03 ms** | 21.98 ms | 19.61 ms | 29.37 ms | 121.62 ms | **1.40x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **12.75 ms** | 16.90 ms | 17.99 ms | 30.66 ms | 80.85 ms | **1.33x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **11.80 ms** | 16.54 ms | 15.62 ms | 26.94 ms | 57.46 ms | **1.32x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **3.42 ms** | 7.15 ms | 4.36 ms | 9.26 ms | 40.68 ms | **1.28x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.0 ns** | 592.4 ns | 1.43 µs | 205.1 ns | 12.77 µs | **51.65x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.7 ns** | 593.0 ns | 1.34 µs | 205.4 ns | 39.20 µs | **54.93x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **12.7 ns** | 71.5 ns | 877.1 ns | 16.5 ns | 8.98 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **660.6 ns** | 3.03 µs | 238.84 µs | 761.8 ns | 90.61 µs | **1.15x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.13 µs** | 23.57 µs | - | 23.49 µs | - | **1.79x** (Arrow (arrow-rs)) |
| sum of i64 column | **1.32 ms** | 2.50 ms | 2.50 ms | 2.56 ms | 2.91 ms | **1.89x** (Polars (Python wheel)) |
| sum of f64 column | **1.36 ms** | 2.97 ms | 2.44 ms | 2.47 ms | 7.34 ms | **1.80x** (Polars (Python wheel)) |
| mean of f64 column | **1.35 ms** | 2.98 ms | 2.76 ms | 2.50 ms | 11.02 ms | **1.86x** (Arrow (arrow-rs)) |
| min and max of f64 column | **2.49 ms** | 6.09 ms | 6.23 ms | 7.91 ms | 20.64 ms | **2.44x** (Polars (Rust crate)) |
| min and max of i64 column | **2.85 ms** | 5.56 ms | 5.81 ms | 5.48 ms | 5.76 ms | **1.92x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **2.45 ms** | 3.08 ms | 12.54 ms | 3.09 ms | 6.61 ms | **1.26x** (Polars (Rust crate)) |
