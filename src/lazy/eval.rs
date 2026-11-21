//! Whole-column evaluation of expressions.
//!
//! This is the general path: every sub-expression produces a full [`Series`].
//! Filters and filtered aggregations normally do not come through here — the
//! executor compiles those to block-wise kernels (see [`super::exec`]) and
//! only falls back to this evaluator for the parts it cannot compile.

use super::dsl::{Expr, LiteralValue, Operator};
use crate::chunked_array::{BooleanChunked, UInt32Chunked, Wide};
use crate::datatypes::{AnyValue, DataType, IdxSize, PlSmallStr};
use crate::error::PolarsResult;
use crate::frame::{AggKind, DataFrame};
use crate::polars_bail;
use crate::series::{ArithOp, IntoSeries, Series};

/// A literal as a one-row series, typed to combine with a column of type `like`.
///
/// Integer literals adopt an integer or float column's type when the value
/// fits; float literals stay `Float32` next to a `Float32` column.
pub(crate) fn literal_series(value: &LiteralValue, like: Option<&DataType>) -> PolarsResult<Series> {
    let name: PlSmallStr = "literal".into();
    Ok(match value {
        LiteralValue::Null => Series::full_null(name, 1, like.unwrap_or(&DataType::Boolean)),
        LiteralValue::Boolean(b) => Series::new(name, [*b]),
        LiteralValue::String(s) => Series::new(name, [s.as_str()]),
        LiteralValue::Float(f) => match like {
            Some(DataType::Float32) => Series::new(name, [*f as f32]),
            _ => Series::new(name, [*f]),
        },
        LiteralValue::Int(i) => {
            let wide = Series::from_any_values(
                name.clone(),
                &[if *i > i64::MAX as i128 { AnyValue::UInt64(*i as u64) } else { AnyValue::Int64(*i as i64) }],
            )?;
            match like {
                Some(dtype) if dtype.is_numeric() => {
                    let cast = wide.cast(dtype)?;
                    // A literal that does not fit the column type keeps its own.
                    if cast.null_count() == 0 { cast } else { wide }
                }
                _ => wide,
            }
        }
    })
}

fn literal_wide(value: &LiteralValue) -> Option<Wide> {
    match value {
        LiteralValue::Int(i) => Some(Wide::Int(*i)),
        LiteralValue::Float(f) => Some(Wide::Float(*f)),
        LiteralValue::Boolean(b) => Some(Wide::Int(*b as i128)),
        _ => None,
    }
}

/// Reduce a column to one value.
pub(crate) fn reduce(s: &Series, kind: AggKind) -> PolarsResult<Series> {
    let name = s.name().clone();
    let dtype = *s.dtype();
    if !kind.supports(&dtype) {
        polars_bail!(InvalidOperation: "`{}` operation not supported for dtype `{}`", kind.suffix(), dtype);
    }
    let from_any = |v: AnyValue<'_>, dtype: &DataType| -> PolarsResult<Series> {
        if v.is_null() {
            return Ok(Series::full_null(name.clone(), 1, dtype));
        }
        Series::from_any_values(name.clone(), &[v])?.cast(dtype)
    };
    let float = |v: Option<f64>| from_any(v.map_or(AnyValue::Null, AnyValue::Float64), &DataType::Float64);
    match kind {
        AggKind::Sum => {
            if dtype.is_float() {
                from_any(AnyValue::Float64(s.sum::<f64>()?), &dtype)
            } else if dtype == DataType::UInt64 {
                from_any(AnyValue::UInt64(s.sum::<u64>()?), &dtype)
            } else {
                from_any(AnyValue::Int64(s.sum::<i64>()?), &DataType::Int64)
            }
        }
        AggKind::Mean => float(s.mean()),
        AggKind::Median => float(s.median()),
        AggKind::Var(ddof) => float(s.var(ddof)),
        AggKind::Std(ddof) => float(s.std(ddof)),
        AggKind::Min => from_any(s.min_any(), &dtype),
        AggKind::Max => from_any(s.max_any(), &dtype),
        AggKind::First => Ok(if s.is_empty() { Series::full_null(name, 1, &dtype) } else { s.slice(0, 1) }),
        AggKind::Last => Ok(if s.is_empty() { Series::full_null(name, 1, &dtype) } else { s.slice(-1, 1) }),
        AggKind::Count => Ok(UInt32Chunked::from_slice(name, &[(s.len() - s.null_count()) as IdxSize]).into_series()),
        AggKind::Len => Ok(UInt32Chunked::from_slice(name, &[s.len() as IdxSize]).into_series()),
        AggKind::NUnique => Ok(UInt32Chunked::from_slice(name, &[s.n_unique()? as IdxSize]).into_series()),
    }
}

/// Repeat a one-row series to `height` rows (no-op if it already has them).
pub(crate) fn broadcast(s: Series, height: usize) -> PolarsResult<Series> {
    if s.len() == height || s.len() != 1 {
        return Ok(s);
    }
    s.take_slice(&vec![0 as IdxSize; height])
}

fn as_bool<'a>(s: &'a Series, what: &str) -> PolarsResult<&'a BooleanChunked> {
    match s.bool() {
        Ok(b) => Ok(b),
        Err(_) => polars_bail!(InvalidOperation: "{} expects boolean operands, got `{}`", what, s.dtype()),
    }
}

fn binary(df: &DataFrame, left: &Expr, op: Operator, right: &Expr) -> PolarsResult<Series> {
    // Evaluate the non-literal side first so that literals can adopt its type.
    let (l, r) = match (left, right) {
        (Expr::Literal(a), Expr::Literal(b)) => (literal_series(a, None)?, literal_series(b, None)?),
        (Expr::Literal(a), _) => {
            let r = eval(df, right)?;
            (literal_series(a, Some(r.dtype()))?, r)
        }
        (_, Expr::Literal(b)) => {
            let l = eval(df, left)?;
            let r = literal_series(b, Some(l.dtype()))?;
            (l, r)
        }
        _ => (eval(df, left)?, eval(df, right)?),
    };
    let name = match left {
        Expr::Literal(_) if !matches!(right, Expr::Literal(_)) => r.name().clone(),
        _ => l.name().clone(),
    };
    Ok(match op {
        Operator::Cmp(cmp) => {
            let mask = match (left, right) {
                // Column against a numeric literal: the scalar kernel, with
                // the literal fitted to the column type mathematically.
                (_, Expr::Literal(v)) if literal_wide(v).is_some() && (l.dtype().is_numeric() || l.dtype().is_bool()) => {
                    l.compare_wide(cmp, literal_wide(v).expect("checked"))?
                }
                (Expr::Literal(v), _) if literal_wide(v).is_some() && (r.dtype().is_numeric() || r.dtype().is_bool()) => {
                    r.compare_wide(cmp.swap_operands(), literal_wide(v).expect("checked"))?
                }
                (_, Expr::Literal(LiteralValue::String(s))) if l.dtype().is_string() => l.compare_str(cmp, s)?,
                (Expr::Literal(LiteralValue::String(s)), _) if r.dtype().is_string() => r.compare_str(cmp.swap_operands(), s)?,
                (_, Expr::Literal(LiteralValue::Null)) | (Expr::Literal(LiteralValue::Null), _) => {
                    BooleanChunked::full_null(name.clone(), l.len().max(r.len()))
                }
                _ => l.compare(cmp, &r)?,
            };
            mask.with_name(name).into_series()
        }
        Operator::Arith(arith) => l.arithmetic(arith, &r)?.with_name(name),
        Operator::And | Operator::Or | Operator::Xor => {
            // A one-row operand adopts the other's length (which may be zero).
            let n = match (l.len(), r.len()) {
                (1, n) | (n, 1) => n,
                (n, _) => n,
            };
            let (l, r) = (broadcast(l, n)?, broadcast(r, n)?);
            let (a, b) = (as_bool(&l, "a logical operator")?, as_bool(&r, "a logical operator")?);
            let out = match op {
                Operator::And => a & b,
                Operator::Or => a | b,
                _ => a ^ b,
            };
            out.with_name(name).into_series()
        }
    })
}

/// Evaluate `expr` over the full height of `df`.
///
/// Aggregations and literals yield a one-row series; the caller broadcasts.
pub(crate) fn eval(df: &DataFrame, expr: &Expr) -> PolarsResult<Series> {
    match expr {
        Expr::Column(name) => df.column(name).cloned(),
        Expr::Wildcard => polars_bail!(InvalidOperation: "the wildcard `*` is only allowed as a top-level `select` expression"),
        Expr::Literal(v) => literal_series(v, None),
        Expr::Alias(e, name) => Ok(eval(df, e)?.with_name(name.clone())),
        Expr::Cast { expr, dtype } => eval(df, expr)?.cast(dtype),
        Expr::Not(e) => {
            let s = eval(df, e)?;
            Ok((!as_bool(&s, "`not`")?).into_series())
        }
        Expr::Neg(e) => eval(df, e)?.arithmetic_scalar(ArithOp::Mul, -1),
        Expr::IsNull(e) => Ok(eval(df, e)?.is_null().into_series()),
        Expr::IsNotNull(e) => Ok(eval(df, e)?.is_not_null().into_series()),
        Expr::Len => Ok(UInt32Chunked::from_slice("len".into(), &[df.height() as IdxSize]).into_series()),
        Expr::Agg { input, kind } => reduce(&eval(df, input)?, *kind),
        Expr::Binary { left, op, right } => binary(df, left, *op, right),
        Expr::Ternary { predicate, truthy, falsy } => {
            let mask = eval(df, predicate)?;
            let mask = as_bool(&mask, "`when`")?.true_mask();
            let (t, f) = match (&**truthy, &**falsy) {
                (Expr::Literal(a), Expr::Literal(b)) => (literal_series(a, None)?, literal_series(b, None)?),
                (Expr::Literal(a), _) => {
                    let f = eval(df, falsy)?;
                    (literal_series(a, Some(f.dtype()))?, f)
                }
                (_, Expr::Literal(b)) => {
                    let t = eval(df, truthy)?;
                    let f = literal_series(b, Some(t.dtype()))?;
                    (t, f)
                }
                _ => (eval(df, truthy)?, eval(df, falsy)?),
            };
            let n = mask.len();
            let name = t.name().clone();
            let common = if t.dtype() == f.dtype() {
                *t.dtype()
            } else {
                match DataType::supertype(t.dtype(), f.dtype()) {
                    Some(d) => d,
                    None => polars_bail!(SchemaMismatch: "`then` and `otherwise` have incompatible types `{}` and `{}`", t.dtype(), f.dtype()),
                }
            };
            let (t, f) = (broadcast(t.cast(&common)?, n)?, broadcast(f.cast(&common)?, n)?);
            // Stack both branches and gather: row `i` from the first half
            // where the mask holds, from the second half elsewhere.
            let mut stacked = t;
            stacked.append(&f)?;
            let idx: Vec<IdxSize> = mask.iter().enumerate().map(|(i, m)| if m { i } else { n + i } as IdxSize).collect();
            Ok(stacked.take_slice(&idx)?.with_name(name))
        }
    }
}

/// Evaluate a predicate to a boolean column (nulls preserved).
pub(crate) fn eval_mask(df: &DataFrame, expr: &Expr) -> PolarsResult<BooleanChunked> {
    let s = eval(df, expr)?;
    let mask = as_bool(&s, "a filter predicate")?.clone();
    if mask.len() == 1 && df.height() != 1 {
        return Ok(match mask.get(0) {
            Some(v) => BooleanChunked::full(mask.name().clone(), v, df.height()),
            None => BooleanChunked::full_null(mask.name().clone(), df.height()),
        });
    }
    Ok(mask)
}

#[cfg(test)]
mod tests {
    use super::super::dsl::{col, len, lit, when};
    use super::*;
    use crate::df;

    fn sample() -> DataFrame {
        df!(
            "a" => [1i32, 2, 3, 4],
            "b" => [Some(0.5f64), None, Some(2.5), Some(3.5)],
            "s" => ["x", "y", "x", "z"],
            "t" => [true, false, true, true],
        )
        .unwrap()
    }

    #[test]
    fn arithmetic_types_follow_the_lazy_rules() {
        let df = sample();
        let e = eval(&df, &(col("a") * 2)).unwrap();
        assert_eq!(e.i32().unwrap().cont_slice().unwrap(), &[2, 4, 6, 8]);
        let f = eval(&df, &(col("a") * 2.5)).unwrap();
        assert_eq!(f.f64().unwrap().cont_slice().unwrap(), &[2.5, 5.0, 7.5, 10.0]);
        let g = eval(&df, &(lit(10) - col("a"))).unwrap();
        assert_eq!((g.name().as_str(), g.i32().unwrap().get(0)), ("a", Some(9)));
        let h = eval(&df, &(col("a") + col("b"))).unwrap();
        assert_eq!(h.f64().unwrap().to_vec(), vec![Some(1.5), None, Some(5.5), Some(7.5)]);
        let big = eval(&df, &(col("a") + 3_000_000_000i64)).unwrap();
        assert_eq!(*big.dtype(), DataType::Int64);
        assert_eq!(eval(&df, &(-col("a"))).unwrap().i32().unwrap().get(0), Some(-1));
        assert_eq!(eval(&df, &(lit(2) + lit(3))).unwrap().i64().unwrap().get(0), Some(5));
        assert!(eval(&df, &(col("s") + 1)).is_err());
        assert_eq!(eval(&df, &col("a").cast(DataType::Float32)).unwrap().f32().unwrap().get(1), Some(2.0));
    }

    #[test]
    fn comparisons_and_logic() {
        let df = sample();
        let m = eval_mask(&df, &col("a").gt(2)).unwrap();
        assert_eq!(m.to_vec(), vec![Some(false), Some(false), Some(true), Some(true)]);
        assert_eq!(eval_mask(&df, &lit(2).lt(col("a"))).unwrap().to_vec(), m.to_vec());
        assert_eq!(eval_mask(&df, &col("a").gt(2.5)).unwrap().num_trues(), 2);
        assert_eq!(eval_mask(&df, &col("b").gt(1.0)).unwrap().to_vec(), vec![Some(false), None, Some(true), Some(true)]);
        assert_eq!(eval_mask(&df, &col("s").eq("x")).unwrap().num_trues(), 2);
        assert_eq!(eval_mask(&df, &lit("y").eq(col("s"))).unwrap().num_trues(), 1);
        assert_eq!(eval_mask(&df, &col("a").gt(col("b"))).unwrap().to_vec(), vec![Some(true), None, Some(true), Some(true)]);
        let both = eval_mask(&df, &col("a").gt(1).and(col("t"))).unwrap();
        assert_eq!(both.to_vec(), vec![Some(false), Some(false), Some(true), Some(true)]);
        assert_eq!(eval_mask(&df, &col("t").not().or(col("a").eq(1))).unwrap().num_trues(), 2);
        assert_eq!(eval_mask(&df, &col("b").is_null()).unwrap().num_trues(), 1);
        assert_eq!(eval_mask(&df, &lit(true)).unwrap().num_trues(), 4);
        assert!(eval_mask(&df, &col("a")).is_err());
        assert!(eval_mask(&df, &col("s").gt(1)).is_err());
        assert_eq!(eval_mask(&df, &col("a").eq(lit(None::<i32>))).unwrap().null_count(), 4);
        assert!(eval(&df, &col("missing")).is_err());
    }

    #[test]
    fn aggregations_and_conditionals() {
        let df = sample();
        assert_eq!(eval(&df, &col("a").sum()).unwrap().i64().unwrap().get(0), Some(10));
        assert_eq!(eval(&df, &col("b").sum()).unwrap().f64().unwrap().get(0), Some(6.5));
        assert_eq!(eval(&df, &col("b").mean()).unwrap().f64().unwrap().get(0), Some(6.5 / 3.0));
        assert_eq!(eval(&df, &col("s").max()).unwrap().str().unwrap().get(0), Some("z"));
        assert_eq!(eval(&df, &col("b").count()).unwrap().u32().unwrap().get(0), Some(3));
        assert_eq!(eval(&df, &col("s").n_unique()).unwrap().u32().unwrap().get(0), Some(3));
        assert_eq!(eval(&df, &len()).unwrap().u32().unwrap().get(0), Some(4));
        assert_eq!(eval(&df, &col("a").last()).unwrap().i32().unwrap().get(0), Some(4));
        assert_eq!(eval(&df, &(col("a").max() - col("a").min())).unwrap().i32().unwrap().get(0), Some(3));
        assert!(eval(&df, &col("s").sum()).is_err());
        let w = eval(&df, &when(col("a").gt(2)).then(col("a")).otherwise(lit(0))).unwrap();
        assert_eq!(w.i32().unwrap().cont_slice().unwrap(), &[0, 0, 3, 4]);
        let w = eval(&df, &when(col("t")).then(lit("yes")).otherwise(col("s"))).unwrap();
        assert_eq!(w.str().unwrap().to_vec(), vec![Some("yes"), Some("y"), Some("yes"), Some("yes")]);
        let w = eval(&df, &when(col("b").gt(1.0)).then(col("b")).otherwise(col("a"))).unwrap();
        assert_eq!(w.f64().unwrap().to_vec(), vec![Some(1.0), Some(2.0), Some(2.5), Some(3.5)]);
    }
}
