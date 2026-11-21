**dataframes is the fastest library in 75 of 75 comparisons** and slower in none. Geometric-mean speed-up over the fastest competitor: **2.49x** (smallest 1.06x, largest 56.72x).

Machine: AMD EPYC 7763 64-Core Processor · 4 threads · Linux 7.0.0-1020-azure · 2026-11-21 04:01 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the best median over the rounds (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **6.86 µs** | 55.62 µs | 21.29 µs | 10.32 µs | 51.98 µs | **1.51x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **11.04 µs** | 59.28 µs | 25.84 µs | 13.49 µs | 189.54 µs | **1.22x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **18.03 µs** | 67.16 µs | 36.16 µs | 22.79 µs | 400.51 µs | **1.26x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **21.12 µs** | 89.86 µs | 62.04 µs | 67.06 µs | 725.30 µs | **2.94x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **21.76 µs** | 90.87 µs | 61.77 µs | 66.11 µs | 1.33 ms | **2.84x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **21.30 µs** | 95.42 µs | 64.91 µs | 64.83 µs | 725.21 µs | **3.04x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **12.62 µs** | 91.11 µs | 56.14 µs | 62.81 µs | 724.40 µs | **4.45x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **12.73 µs** | 100.02 µs | 56.58 µs | 71.56 µs | 722.02 µs | **4.44x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **43.04 µs** | 129.12 µs | 195.02 µs | 75.96 µs | 420.87 µs | **1.76x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **76.59 µs** | 244.66 µs | 316.76 µs | 188.03 µs | 1.83 ms | **2.45x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **75.43 µs** | 177.46 µs | 236.36 µs | 165.12 µs | 2.01 ms | **2.19x** (Arrow (arrow-rs)) |
| filter 6-column frame by expression, 50% | **78.62 µs** | 239.59 µs | 358.32 µs | 187.92 µs | 1.83 ms | **2.39x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **60.67 µs** | 250.90 µs | 362.25 µs | 125.01 µs | 674.76 µs | **2.06x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **32.66 µs** | 239.55 µs | 273.54 µs | 66.48 µs | 450.09 µs | **2.04x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **4.5 ns** | 649.5 ns | 1.43 µs | 219.1 ns | 13.10 µs | **48.44x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 632.3 ns | 1.37 µs | 212.8 ns | 39.38 µs | **56.02x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **14.6 ns** | 75.1 ns | 904.3 ns | 19.2 ns | 9.07 µs | **1.32x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **633.5 ns** | 3.40 µs | 234.86 µs | 760.6 ns | 92.46 µs | **1.20x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.26 µs** | 23.19 µs | - | 23.15 µs | - | **1.75x** (Arrow (arrow-rs)) |
| sum of i64 column | **4.42 µs** | 9.43 µs | 10.67 µs | 9.39 µs | 22.67 µs | **2.13x** (Arrow (arrow-rs)) |
| sum of f64 column | **4.50 µs** | 10.89 µs | 11.07 µs | 11.66 µs | 52.28 µs | **2.42x** (Polars (Rust crate)) |
| mean of f64 column | **4.50 µs** | 11.45 µs | 10.82 µs | 11.65 µs | 85.48 µs | **2.40x** (Polars (Python wheel)) |
| min and max of f64 column | **8.89 µs** | 21.13 µs | 23.73 µs | 58.38 µs | 151.18 µs | **2.38x** (Polars (Rust crate)) |
| min and max of i64 column | **9.15 µs** | 19.69 µs | 32.73 µs | 18.94 µs | 48.30 µs | **2.07x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **5.94 µs** | 10.42 µs | 62.10 µs | 10.79 µs | 33.30 µs | **1.75x** (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **62.75 µs** | 561.20 µs | 118.70 µs | 100.41 µs | 162.44 µs | **1.60x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **89.16 µs** | 595.71 µs | 156.74 µs | 156.09 µs | 882.14 µs | **1.75x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **156.28 µs** | 722.90 µs | 284.16 µs | 296.05 µs | 2.92 ms | **1.82x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **189.22 µs** | 903.62 µs | 510.49 µs | 672.28 µs | 9.65 ms | **2.70x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **191.49 µs** | 901.19 µs | 509.24 µs | 1.22 ms | 5.42 ms | **2.66x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **188.93 µs** | 951.12 µs | 557.05 µs | 649.14 µs | 9.70 ms | **2.95x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **106.48 µs** | 915.13 µs | 454.52 µs | 633.22 µs | 9.57 ms | **4.27x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **106.24 µs** | 1.01 ms | 464.01 µs | 719.84 µs | 9.54 ms | **4.37x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **727.19 µs** | 1.27 ms | 1.11 ms | 1.44 ms | 2.64 ms | **1.52x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **1.22 ms** | 1.99 ms | 1.81 ms | 2.76 ms | 12.59 ms | **1.48x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **1.25 ms** | 1.69 ms | 1.60 ms | 2.56 ms | 14.28 ms | **1.28x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **1.19 ms** | 1.55 ms | 1.66 ms | 2.79 ms | 12.41 ms | **1.30x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **1.13 ms** | 1.57 ms | 1.49 ms | 2.27 ms | 4.12 ms | **1.33x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **239.58 µs** | 901.52 µs | 633.71 µs | 667.90 µs | 6.03 ms | **2.65x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.6 ns** | 654.3 ns | 1.46 µs | 215.0 ns | 13.04 µs | **47.14x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 634.7 ns | 1.38 µs | 206.2 ns | 38.91 µs | **54.25x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **14.6 ns** | 74.4 ns | 904.7 ns | 19.2 ns | 9.17 µs | **1.32x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **650.2 ns** | 3.39 µs | 231.88 µs | 775.4 ns | 92.07 µs | **1.19x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.26 µs** | 23.22 µs | - | 23.16 µs | - | **1.75x** (Arrow (arrow-rs)) |
| sum of i64 column | **53.48 µs** | 96.67 µs | 99.11 µs | 96.62 µs | 113.65 µs | **1.81x** (Arrow (arrow-rs)) |
| sum of f64 column | **54.23 µs** | 114.69 µs | 105.64 µs | 116.37 µs | 359.59 µs | **1.95x** (Polars (Python wheel)) |
| mean of f64 column | **54.40 µs** | 114.97 µs | 107.34 µs | 116.38 µs | 702.82 µs | **1.97x** (Polars (Python wheel)) |
| min and max of f64 column | **108.05 µs** | 211.44 µs | 226.21 µs | 591.54 µs | 1.18 ms | **1.96x** (Polars (Rust crate)) |
| min and max of i64 column | **107.31 µs** | 196.63 µs | 314.38 µs | 194.19 µs | 225.07 µs | **1.81x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **53.21 µs** | 104.44 µs | 510.74 µs | 104.44 µs | 286.70 µs | **1.96x** (Arrow (arrow-rs)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.48 ms** | 5.79 ms | 2.77 ms | 2.95 ms | 3.16 ms | **1.88x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **1.67 ms** | 7.18 ms | 4.21 ms | 4.12 ms | 11.66 ms | **2.47x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **2.38 ms** | 8.64 ms | 5.78 ms | 5.91 ms | 35.31 ms | **2.43x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **3.48 ms** | 10.02 ms | 7.49 ms | 8.80 ms | 70.61 ms | **2.15x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **4.59 ms** | 10.44 ms | 7.86 ms | 14.20 ms | 48.12 ms | **1.71x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **3.53 ms** | 10.61 ms | 8.04 ms | 8.89 ms | 70.37 ms | **2.28x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **1.47 ms** | 9.33 ms | 5.28 ms | 6.97 ms | 70.36 ms | **3.58x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **1.53 ms** | 10.25 ms | 5.47 ms | 7.45 ms | 69.06 ms | **3.58x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **11.65 ms** | 16.15 ms | 13.61 ms | 18.88 ms | 40.21 ms | **1.17x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **15.68 ms** | 21.62 ms | 18.91 ms | 31.13 ms | 79.78 ms | **1.21x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **15.91 ms** | 21.57 ms | 19.84 ms | 29.28 ms | 118.84 ms | **1.25x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **15.21 ms** | 16.55 ms | 17.18 ms | 31.02 ms | 80.12 ms | **1.09x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **13.17 ms** | 15.95 ms | 15.21 ms | 26.38 ms | 57.94 ms | **1.15x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **3.94 ms** | 7.07 ms | 4.18 ms | 9.47 ms | 39.10 ms | **1.06x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.4 ns** | 636.0 ns | 1.41 µs | 206.8 ns | 13.02 µs | **46.70x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.8 ns** | 631.2 ns | 1.34 µs | 216.2 ns | 40.09 µs | **56.72x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **14.6 ns** | 73.4 ns | 893.2 ns | 19.5 ns | 9.09 µs | **1.34x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **656.7 ns** | 3.41 µs | 233.08 µs | 777.4 ns | 92.03 µs | **1.18x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.26 µs** | 24.69 µs | - | 23.77 µs | - | **1.79x** (Arrow (arrow-rs)) |
| sum of i64 column | **1.31 ms** | 2.29 ms | 2.42 ms | 2.29 ms | 2.93 ms | **1.75x** (Polars (Rust crate)) |
| sum of f64 column | **1.31 ms** | 2.93 ms | 2.47 ms | 2.29 ms | 7.16 ms | **1.74x** (Arrow (arrow-rs)) |
| mean of f64 column | **1.33 ms** | 2.91 ms | 2.47 ms | 2.31 ms | 10.51 ms | **1.73x** (Arrow (arrow-rs)) |
| min and max of f64 column | **2.69 ms** | 5.94 ms | 6.11 ms | 7.19 ms | 20.22 ms | **2.20x** (Polars (Rust crate)) |
| min and max of i64 column | **2.97 ms** | 5.50 ms | 5.63 ms | 5.37 ms | 5.72 ms | **1.81x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **2.09 ms** | 2.77 ms | 12.32 ms | 2.75 ms | 5.82 ms | **1.32x** (Arrow (arrow-rs)) |
