//! Text rendering of frames, in the layout Polars uses.

use std::fmt;

use super::DataFrame;
use crate::datatypes::{AnyValue, fmt_float};

/// Rows shown before a frame is abbreviated to its head and tail.
const MAX_ROWS: usize = 10;
/// Columns shown before a frame is abbreviated to its first and last columns.
const MAX_COLS: usize = 8;
/// Longest cell, in characters, before it is truncated.
const MAX_CELL: usize = 32;

fn cell(value: &AnyValue<'_>) -> String {
    let text = match value {
        AnyValue::Null => "null".to_string(),
        AnyValue::String(s) => (*s).to_string(),
        AnyValue::StringOwned(s) => s.to_string(),
        AnyValue::Float32(v) => fmt_float(*v as f64),
        AnyValue::Float64(v) => fmt_float(*v),
        other => other.to_string(),
    };
    if text.chars().count() > MAX_CELL {
        let mut cut: String = text.chars().take(MAX_CELL - 1).collect();
        cut.push('…');
        cut
    } else {
        text
    }
}

impl fmt::Display for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (height, width) = self.shape();
        writeln!(f, "shape: ({height}, {width})")?;
        if width == 0 {
            return write!(f, "┌┐\n╞╡\n└┘");
        }

        // Which columns and rows are visible (`None` marks an ellipsis).
        let cols: Vec<Option<usize>> = if width <= MAX_COLS {
            (0..width).map(Some).collect()
        } else {
            (0..MAX_COLS / 2).map(Some).chain([None]).chain((width - MAX_COLS / 2..width).map(Some)).collect()
        };
        let rows: Vec<Option<usize>> = if height <= MAX_ROWS {
            (0..height).map(Some).collect()
        } else {
            (0..MAX_ROWS / 2).map(Some).chain([None]).chain((height - MAX_ROWS / 2..height).map(Some)).collect()
        };

        // One vector of lines per visible column: name, "---", dtype, cells.
        let table: Vec<Vec<String>> = cols
            .iter()
            .map(|c| match c {
                None => std::iter::repeat_n("…".to_string(), 3 + rows.len()).collect(),
                Some(c) => {
                    let col = &self.columns()[*c];
                    let mut lines = vec![col.name().to_string(), "---".to_string(), col.dtype().to_string()];
                    lines.extend(rows.iter().map(|r| match r {
                        None => "…".to_string(),
                        Some(r) => cell(&col.get_any(*r)),
                    }));
                    lines
                }
            })
            .collect();
        let widths: Vec<usize> =
            table.iter().map(|lines| lines.iter().map(|l| l.chars().count()).max().unwrap_or(0)).collect();

        let rule = |f: &mut fmt::Formatter<'_>, left: &str, fill: &str, mid: &str, right: &str| -> fmt::Result {
            f.write_str(left)?;
            for (i, w) in widths.iter().enumerate() {
                if i > 0 {
                    f.write_str(mid)?;
                }
                f.write_str(&fill.repeat(w + 2))?;
            }
            f.write_str(right)
        };
        let line = |f: &mut fmt::Formatter<'_>, row: usize| -> fmt::Result {
            f.write_str("│")?;
            for (i, (lines, w)) in table.iter().zip(&widths).enumerate() {
                if i > 0 {
                    f.write_str("┆")?;
                }
                let text = &lines[row];
                write!(f, " {text}{} ", " ".repeat(w - text.chars().count()))?;
            }
            f.write_str("│")
        };

        rule(f, "┌", "─", "┬", "┐")?;
        writeln!(f)?;
        for header in 0..3 {
            line(f, header)?;
            writeln!(f)?;
        }
        rule(f, "╞", "═", "╪", "╡")?;
        writeln!(f)?;
        for r in 0..rows.len() {
            line(f, 3 + r)?;
            writeln!(f)?;
        }
        rule(f, "└", "─", "┴", "┘")
    }
}

impl fmt::Debug for DataFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests {
    use crate::df;

    #[test]
    fn renders_like_polars() {
        let df = df!("a" => [Some(1i64), None, Some(3)], "b" => [1.0, 2.5, 3.0], "name" => ["x", "yy", "zzz"]).unwrap();
        let expected = "\
shape: (3, 3)
┌──────┬─────┬──────┐
│ a    ┆ b   ┆ name │
│ ---  ┆ --- ┆ ---  │
│ i64  ┆ f64 ┆ str  │
╞══════╪═════╪══════╡
│ 1    ┆ 1.0 ┆ x    │
│ null ┆ 2.5 ┆ yy   │
│ 3    ┆ 3.0 ┆ zzz  │
└──────┴─────┴──────┘";
        assert_eq!(df.to_string(), expected);
    }

    #[test]
    fn abbreviates_long_and_wide_frames() {
        let tall = df!("i" => (0..100).collect::<Vec<i32>>()).unwrap();
        let text = tall.to_string();
        assert!(text.starts_with("shape: (100, 1)"));
        assert_eq!(text.lines().filter(|l| l.contains('…')).count(), 1);
        assert!(text.contains("│ 4   │") && text.contains("│ 95  │") && !text.contains("│ 50  │"));

        let cols: Vec<crate::series::Series> =
            (0..12).map(|i| crate::series::Series::new(format!("c{i}").into(), [i])).collect();
        let wide = crate::frame::DataFrame::new(1, cols).unwrap();
        let text = wide.to_string();
        assert!(text.contains("c3") && text.contains("c8") && !text.contains("c5"));
        assert!(text.contains("┆ … ┆"));
        assert_eq!(crate::frame::DataFrame::empty().to_string(), "shape: (0, 0)\n┌┐\n╞╡\n└┘");
        let long = df!("s" => ["a string that is clearly longer than thirty-two characters"]).unwrap();
        assert!(long.to_string().contains("a string that is clearly longer…"));
    }
}
