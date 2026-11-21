**dataframes is the fastest library in 75 of 75 comparisons** and slower in none. Geometric-mean speed-up over the fastest competitor: **2.45x** (smallest 1.09x, largest 57.59x).

Machine: AMD EPYC 7763 64-Core Processor · 4 threads · Linux 7.0.0-1020-azure · 2026-11-21 04:21 UTC  
Versions: Polars (Rust crate) 0.55.2 · Arrow (arrow-rs) 60.1.0 · Polars (Python wheel) 2.1.0 · pandas 3.1.1

Times are the best median over the rounds (lower is better). The last column is the fastest competitor's time divided by ours.

### 100K rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **6.93 µs** | 55.02 µs | 20.35 µs | 12.66 µs | 52.24 µs | **1.83x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **11.20 µs** | 58.61 µs | 25.42 µs | 13.56 µs | 190.48 µs | **1.21x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **18.17 µs** | 70.34 µs | 35.92 µs | 27.83 µs | 395.83 µs | **1.53x** (Arrow (arrow-rs)) |
| filter i64 column, 50% selected | **21.39 µs** | 89.33 µs | 62.07 µs | 67.14 µs | 727.82 µs | **2.90x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **22.06 µs** | 90.08 µs | 61.97 µs | 66.05 µs | 1.35 ms | **2.81x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **21.53 µs** | 103.29 µs | 66.92 µs | 64.83 µs | 727.40 µs | **3.01x** (Arrow (arrow-rs)) |
| filter i32 column, 50% selected | **12.80 µs** | 90.88 µs | 55.67 µs | 63.00 µs | 727.15 µs | **4.35x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **12.81 µs** | 95.74 µs | 56.57 µs | 71.88 µs | 724.26 µs | **4.42x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **44.37 µs** | 137.47 µs | 192.83 µs | 76.02 µs | 422.82 µs | **1.71x** (Arrow (arrow-rs)) |
| filter 6-column frame by mask, 50% | **76.75 µs** | 242.60 µs | 314.24 µs | 188.63 µs | 1.83 ms | **2.46x** (Arrow (arrow-rs)) |
| filter frame with a string column, 50% | **75.76 µs** | 174.68 µs | 236.74 µs | 189.73 µs | 2.04 ms | **2.31x** (Polars (Rust crate)) |
| filter 6-column frame by expression, 50% | **78.50 µs** | 241.86 µs | 358.25 µs | 188.51 µs | 1.85 ms | **2.40x** (Arrow (arrow-rs)) |
| filter frame by `a < x & b < y`, 25% | **62.20 µs** | 253.62 µs | 362.21 µs | 125.89 µs | 687.61 µs | **2.02x** (Arrow (arrow-rs)) |
| sum of one column over filtered rows | **32.75 µs** | 234.59 µs | 277.92 µs | 66.60 µs | 446.96 µs | **2.03x** (Arrow (arrow-rs)) |
| slice 6-column frame (zero-copy) | **4.6 ns** | 614.4 ns | 1.42 µs | 204.7 ns | 13.02 µs | **44.52x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.7 ns** | 628.1 ns | 1.37 µs | 215.1 ns | 39.67 µs | **57.59x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **12.7 ns** | 72.3 ns | 879.6 ns | 16.4 ns | 9.00 µs | **1.29x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **639.2 ns** | 3.02 µs | 231.66 µs | 749.8 ns | 91.57 µs | **1.17x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.13 µs** | 23.48 µs | - | 23.49 µs | - | **1.79x** (Polars (Rust crate)) |
| sum of i64 column | **4.45 µs** | 9.47 µs | 10.58 µs | 12.10 µs | 22.50 µs | **2.13x** (Polars (Rust crate)) |
| sum of f64 column | **4.51 µs** | 11.88 µs | 13.07 µs | 11.66 µs | 53.96 µs | **2.59x** (Arrow (arrow-rs)) |
| mean of f64 column | **4.52 µs** | 10.87 µs | 10.89 µs | 12.08 µs | 85.54 µs | **2.40x** (Polars (Rust crate)) |
| min and max of f64 column | **8.94 µs** | 20.68 µs | 23.42 µs | 58.37 µs | 151.18 µs | **2.31x** (Polars (Rust crate)) |
| min and max of i64 column | **9.21 µs** | 19.69 µs | 32.57 µs | 18.93 µs | 49.36 µs | **2.06x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **5.99 µs** | 10.45 µs | 61.23 µs | 10.89 µs | 27.96 µs | **1.74x** (Polars (Rust crate)) |

### 1M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **62.76 µs** | 555.25 µs | 145.00 µs | 100.77 µs | 188.59 µs | **1.61x** (Arrow (arrow-rs)) |
| filter i64 column, 1% selected | **89.23 µs** | 589.45 µs | 158.16 µs | 146.11 µs | 884.28 µs | **1.64x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **156.15 µs** | 717.88 µs | 282.07 µs | 294.79 µs | 2.94 ms | **1.81x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **189.62 µs** | 896.23 µs | 508.76 µs | 672.56 µs | 9.80 ms | **2.68x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **190.85 µs** | 896.69 µs | 511.21 µs | 1.21 ms | 5.57 ms | **2.68x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **189.97 µs** | 1.03 ms | 560.33 µs | 651.33 µs | 9.86 ms | **2.95x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **107.01 µs** | 910.22 µs | 452.93 µs | 633.59 µs | 9.66 ms | **4.23x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **106.28 µs** | 959.70 µs | 463.16 µs | 721.41 µs | 9.59 ms | **4.36x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **786.90 µs** | 1.29 ms | 1.15 ms | 1.55 ms | 2.77 ms | **1.46x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **1.31 ms** | 2.02 ms | 1.83 ms | 2.79 ms | 12.48 ms | **1.40x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **1.28 ms** | 1.85 ms | 1.74 ms | 2.62 ms | 14.35 ms | **1.36x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **1.29 ms** | 1.58 ms | 1.69 ms | 2.81 ms | 12.47 ms | **1.23x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **1.22 ms** | 1.67 ms | 1.54 ms | 2.39 ms | 4.22 ms | **1.26x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **246.99 µs** | 905.88 µs | 632.57 µs | 685.50 µs | 6.35 ms | **2.56x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.5 ns** | 632.8 ns | 1.42 µs | 205.4 ns | 13.00 µs | **45.65x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.7 ns** | 615.4 ns | 1.33 µs | 205.4 ns | 38.83 µs | **54.93x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **12.7 ns** | 72.3 ns | 905.7 ns | 16.4 ns | 9.31 µs | **1.29x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **653.4 ns** | 3.04 µs | 224.31 µs | 769.0 ns | 91.96 µs | **1.18x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.13 µs** | 23.57 µs | - | 23.52 µs | - | **1.79x** (Arrow (arrow-rs)) |
| sum of i64 column | **53.90 µs** | 97.05 µs | 99.40 µs | 97.05 µs | 113.75 µs | **1.80x** (Polars (Rust crate)) |
| sum of f64 column | **55.14 µs** | 113.24 µs | 107.25 µs | 116.44 µs | 364.48 µs | **1.94x** (Polars (Python wheel)) |
| mean of f64 column | **54.84 µs** | 111.68 µs | 105.62 µs | 116.93 µs | 694.02 µs | **1.93x** (Polars (Python wheel)) |
| min and max of f64 column | **109.11 µs** | 208.78 µs | 232.16 µs | 592.43 µs | 1.25 ms | **1.91x** (Polars (Rust crate)) |
| min and max of i64 column | **108.50 µs** | 196.59 µs | 314.95 µs | 194.90 µs | 225.96 µs | **1.80x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **53.33 µs** | 104.74 µs | 554.12 µs | 106.62 µs | 290.39 µs | **1.96x** (Polars (Rust crate)) |

### 10M rows

| operation | dataframes | Polars (Rust crate) | Polars (Python wheel) | Arrow (arrow-rs) | pandas | vs fastest competitor |
|---|---:|---:|---:|---:|---:|---:|
| compare i64 column to scalar -> mask | **1.63 ms** | 5.73 ms | 2.82 ms | 2.93 ms | 3.18 ms | **1.73x** (Polars (Python wheel)) |
| filter i64 column, 1% selected | **1.81 ms** | 7.26 ms | 4.36 ms | 4.31 ms | 12.63 ms | **2.38x** (Arrow (arrow-rs)) |
| filter i64 column, 10% selected | **2.53 ms** | 8.69 ms | 5.83 ms | 6.09 ms | 34.80 ms | **2.30x** (Polars (Python wheel)) |
| filter i64 column, 50% selected | **3.86 ms** | 10.10 ms | 7.48 ms | 8.83 ms | 71.17 ms | **1.94x** (Polars (Python wheel)) |
| filter i64 column, 90% selected | **4.86 ms** | 10.57 ms | 8.08 ms | 14.04 ms | 49.00 ms | **1.66x** (Polars (Python wheel)) |
| filter f64 column, 50% selected | **3.76 ms** | 11.40 ms | 8.11 ms | 8.86 ms | 70.49 ms | **2.15x** (Polars (Python wheel)) |
| filter i32 column, 50% selected | **1.52 ms** | 9.27 ms | 5.27 ms | 7.00 ms | 70.77 ms | **3.48x** (Polars (Python wheel)) |
| filter f32 column, 50% selected | **1.57 ms** | 9.79 ms | 5.53 ms | 7.45 ms | 69.73 ms | **3.53x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 10% | **12.81 ms** | 16.89 ms | 14.39 ms | 19.59 ms | 40.07 ms | **1.12x** (Polars (Python wheel)) |
| filter 6-column frame by mask, 50% | **17.00 ms** | 22.10 ms | 19.73 ms | 32.23 ms | 78.74 ms | **1.16x** (Polars (Python wheel)) |
| filter frame with a string column, 50% | **17.31 ms** | 22.60 ms | 20.94 ms | 30.43 ms | 120.31 ms | **1.21x** (Polars (Python wheel)) |
| filter 6-column frame by expression, 50% | **16.19 ms** | 17.66 ms | 18.05 ms | 32.46 ms | 81.31 ms | **1.09x** (Polars (Rust crate)) |
| filter frame by `a < x & b < y`, 25% | **14.33 ms** | 16.73 ms | 15.90 ms | 27.40 ms | 57.80 ms | **1.11x** (Polars (Python wheel)) |
| sum of one column over filtered rows | **4.20 ms** | 7.12 ms | 4.58 ms | 9.59 ms | 40.75 ms | **1.09x** (Polars (Python wheel)) |
| slice 6-column frame (zero-copy) | **4.5 ns** | 625.2 ns | 1.41 µs | 204.9 ns | 13.06 µs | **45.63x** (Arrow (arrow-rs)) |
| head(10) of 6-column frame | **3.7 ns** | 632.5 ns | 1.35 µs | 205.1 ns | 39.70 µs | **54.87x** (Arrow (arrow-rs)) |
| slice one column (zero-copy) | **12.7 ns** | 71.8 ns | 884.1 ns | 16.5 ns | 9.26 µs | **1.30x** (Arrow (arrow-rs)) |
| 1000 random point look-ups in a column | **659.9 ns** | 3.02 µs | 227.41 µs | 752.6 ns | 91.41 µs | **1.14x** (Arrow (arrow-rs)) |
| 1000 random cell look-ups by column name | **13.13 µs** | 23.52 µs | - | 23.52 µs | - | **1.79x** (Polars (Rust crate)) |
| sum of i64 column | **1.48 ms** | 2.46 ms | 2.55 ms | 2.46 ms | 3.00 ms | **1.66x** (Arrow (arrow-rs)) |
| sum of f64 column | **1.47 ms** | 3.00 ms | 2.60 ms | 2.50 ms | 7.48 ms | **1.70x** (Arrow (arrow-rs)) |
| mean of f64 column | **1.50 ms** | 3.05 ms | 2.62 ms | 2.49 ms | 10.62 ms | **1.66x** (Arrow (arrow-rs)) |
| min and max of f64 column | **3.04 ms** | 6.16 ms | 6.34 ms | 7.39 ms | 20.50 ms | **2.03x** (Polars (Rust crate)) |
| min and max of i64 column | **3.16 ms** | 5.61 ms | 5.87 ms | 5.60 ms | 5.85 ms | **1.77x** (Arrow (arrow-rs)) |
| in-place update x = x*c + d | **2.42 ms** | 2.91 ms | 13.23 ms | 2.94 ms | 6.34 ms | **1.20x** (Polars (Rust crate)) |
