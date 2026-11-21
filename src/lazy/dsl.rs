//! The expression language: [`Expr`], [`col`], [`lit`], [`when`].

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Not, Rem, Sub};

use crate::datatypes::{DataType, PlSmallStr};
use crate::frame::AggKind;
use crate::kernels::CmpOp;
use crate::series::ArithOp;

/// A constant inside an expression.
#[derive(Clone, Debug, PartialEq)]
pub enum LiteralValue {
    /// A missing value.
    Null,
    /// A boolean.
    Boolean(bool),
    /// An integer of unspecified width: it adopts the type of the column it
    /// meets (like Polars' "dyn int" literals).
    Int(i128),
    /// A float.
    Float(f64),
    /// A string.
    String(PlSmallStr),
}

/// A binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operator {
    /// A comparison.
    Cmp(CmpOp),
    /// An arithmetic operation.
    Arith(ArithOp),
    /// Logical AND (Kleene).
    And,
    /// Logical OR (Kleene).
    Or,
    /// Logical XOR.
    Xor,
}

/// Whether the ends of an interval belong to it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClosedInterval {
    /// `lower <= x <= upper`
    #[default]
    Both,
    /// `lower <= x < upper`
    Left,
    /// `lower < x <= upper`
    Right,
    /// `lower < x < upper`
    None,
}

/// A computation over the columns of a frame, built with [`col`], [`lit`] and
/// the methods below, and run by a [`LazyFrame`](super::LazyFrame) or by the
/// expression-taking methods of [`DataFrame`](crate::frame::DataFrame).
///
/// ```
/// use dataframes::prelude::*;
/// let df = df!("a" => [1, 2, 3, 4], "b" => [1.0, 0.5, 2.0, 0.1])?;
/// let out = df.lazy().filter(col("a").gt(lit(1)).and(col("b").lt(1.0))).collect()?;
/// assert_eq!(out.column("a")?.i32()?.cont_slice()?, &[2, 4]);
/// # Ok::<(), PolarsError>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A column of the input frame.
    Column(PlSmallStr),
    /// Every column of the input frame (only valid in `select`).
    Wildcard,
    /// A constant.
    Literal(LiteralValue),
    /// `left op right`.
    Binary {
        /// Left operand.
        left: Box<Expr>,
        /// Operator.
        op: Operator,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Logical negation.
    Not(Box<Expr>),
    /// Arithmetic negation.
    Neg(Box<Expr>),
    /// `true` where the input is null.
    IsNull(Box<Expr>),
    /// `true` where the input is not null.
    IsNotNull(Box<Expr>),
    /// An aggregation of the input.
    Agg {
        /// What is aggregated.
        input: Box<Expr>,
        /// How.
        kind: AggKind,
    },
    /// The number of rows (`pl.len()`).
    Len,
    /// The input under another name.
    Alias(Box<Expr>, PlSmallStr),
    /// The input converted to another type.
    Cast {
        /// What is converted.
        expr: Box<Expr>,
        /// Target type.
        dtype: DataType,
    },
    /// `truthy` where `predicate` holds, `falsy` elsewhere.
    Ternary {
        /// The condition.
        predicate: Box<Expr>,
        /// Value where the condition is true.
        truthy: Box<Expr>,
        /// Value where the condition is false or null.
        falsy: Box<Expr>,
    },
}

/// The column called `name`. `col("*")` selects every column.
pub fn col(name: impl Into<PlSmallStr>) -> Expr {
    let name = name.into();
    if name == "*" { Expr::Wildcard } else { Expr::Column(name) }
}

/// Several columns at once: `cols(["a", "b"])`.
pub fn cols<I, S>(names: I) -> Vec<Expr>
where
    I: IntoIterator<Item = S>,
    S: Into<PlSmallStr>,
{
    names.into_iter().map(col).collect()
}

/// Every column of the frame.
pub fn all() -> Expr {
    Expr::Wildcard
}

/// The number of rows.
pub fn len() -> Expr {
    Expr::Len
}

/// A Rust value usable as an expression literal.
pub trait Literal {
    /// Wrap as a literal expression.
    fn lit(self) -> Expr;
}

/// A constant expression.
pub fn lit<L: Literal>(value: L) -> Expr {
    value.lit()
}

macro_rules! literal_impls {
    (int: $($i:ty),*; float: $($f:ty),*) => {
        $(
            impl Literal for $i {
                fn lit(self) -> Expr { Expr::Literal(LiteralValue::Int(self as i128)) }
            }
            impl From<$i> for Expr {
                fn from(v: $i) -> Expr { v.lit() }
            }
        )*
        $(
            impl Literal for $f {
                fn lit(self) -> Expr { Expr::Literal(LiteralValue::Float(self as f64)) }
            }
            impl From<$f> for Expr {
                fn from(v: $f) -> Expr { v.lit() }
            }
        )*
    };
}
literal_impls!(int: i8, i16, i32, i64, isize, u8, u16, u32, u64, usize; float: f32, f64);

impl Literal for bool {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::Boolean(self))
    }
}

impl From<bool> for Expr {
    fn from(v: bool) -> Expr {
        v.lit()
    }
}

impl Literal for &str {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::String(self.into()))
    }
}

impl From<&str> for Expr {
    /// A string literal (use [`col`] for a column reference).
    fn from(v: &str) -> Expr {
        v.lit()
    }
}

impl Literal for String {
    fn lit(self) -> Expr {
        Expr::Literal(LiteralValue::String(self.into()))
    }
}

impl From<String> for Expr {
    fn from(v: String) -> Expr {
        v.lit()
    }
}

impl<L: Literal> Literal for Option<L> {
    fn lit(self) -> Expr {
        self.map_or(Expr::Literal(LiteralValue::Null), Literal::lit)
    }
}

impl Expr {
    fn binary(self, op: Operator, rhs: impl Into<Expr>) -> Expr {
        Expr::Binary { left: Box::new(self), op, right: Box::new(rhs.into()) }
    }

    fn agg(self, kind: AggKind) -> Expr {
        Expr::Agg { input: Box::new(self), kind }
    }

    /// Rename the output.
    pub fn alias(self, name: impl Into<PlSmallStr>) -> Expr {
        Expr::Alias(Box::new(self), name.into())
    }

    /// `self == rhs`
    pub fn eq(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Cmp(CmpOp::Eq), rhs)
    }

    /// `self != rhs`
    pub fn neq(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Cmp(CmpOp::Ne), rhs)
    }

    /// `self < rhs`
    pub fn lt(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Cmp(CmpOp::Lt), rhs)
    }

    /// `self <= rhs`
    pub fn lt_eq(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Cmp(CmpOp::Le), rhs)
    }

    /// `self > rhs`
    pub fn gt(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Cmp(CmpOp::Gt), rhs)
    }

    /// `self >= rhs`
    pub fn gt_eq(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Cmp(CmpOp::Ge), rhs)
    }

    /// Logical AND.
    pub fn and(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::And, rhs)
    }

    /// Logical OR.
    pub fn or(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Or, rhs)
    }

    /// Logical XOR.
    pub fn xor(self, rhs: impl Into<Expr>) -> Expr {
        self.binary(Operator::Xor, rhs)
    }

    /// Logical NOT.
    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }

    /// `true` where the value is null.
    pub fn is_null(self) -> Expr {
        Expr::IsNull(Box::new(self))
    }

    /// `true` where the value is not null.
    pub fn is_not_null(self) -> Expr {
        Expr::IsNotNull(Box::new(self))
    }

    /// `lower <= self <= upper`, with the ends included as `closed` says.
    pub fn is_between(self, lower: impl Into<Expr>, upper: impl Into<Expr>, closed: ClosedInterval) -> Expr {
        let lo = match closed {
            ClosedInterval::Both | ClosedInterval::Left => self.clone().gt_eq(lower),
            _ => self.clone().gt(lower),
        };
        let hi = match closed {
            ClosedInterval::Both | ClosedInterval::Right => self.lt_eq(upper),
            _ => self.lt(upper),
        };
        lo.and(hi)
    }

    /// `true` where the value equals one of `values`.
    pub fn is_in<I, E>(self, values: I) -> Expr
    where
        I: IntoIterator<Item = E>,
        E: Into<Expr>,
    {
        values
            .into_iter()
            .map(|v| self.clone().eq(v))
            .reduce(Expr::or)
            .unwrap_or(Expr::Literal(LiteralValue::Boolean(false)))
    }

    /// Convert to another type.
    pub fn cast(self, dtype: DataType) -> Expr {
        Expr::Cast { expr: Box::new(self), dtype }
    }

    /// Sum.
    pub fn sum(self) -> Expr {
        self.agg(AggKind::Sum)
    }

    /// Mean.
    pub fn mean(self) -> Expr {
        self.agg(AggKind::Mean)
    }

    /// Minimum.
    pub fn min(self) -> Expr {
        self.agg(AggKind::Min)
    }

    /// Maximum.
    pub fn max(self) -> Expr {
        self.agg(AggKind::Max)
    }

    /// First value.
    pub fn first(self) -> Expr {
        self.agg(AggKind::First)
    }

    /// Last value.
    pub fn last(self) -> Expr {
        self.agg(AggKind::Last)
    }

    /// Number of non-null values.
    pub fn count(self) -> Expr {
        self.agg(AggKind::Count)
    }

    /// Number of rows (including nulls).
    pub fn len(self) -> Expr {
        self.agg(AggKind::Len)
    }

    /// Number of distinct values.
    pub fn n_unique(self) -> Expr {
        self.agg(AggKind::NUnique)
    }

    /// Median.
    pub fn median(self) -> Expr {
        self.agg(AggKind::Median)
    }

    /// Standard deviation.
    pub fn std(self, ddof: u8) -> Expr {
        self.agg(AggKind::Std(ddof))
    }

    /// Variance.
    pub fn var(self, ddof: u8) -> Expr {
        self.agg(AggKind::Var(ddof))
    }

    /// The name of the column this expression produces: an explicit alias,
    /// else the left-most column it reads, else `literal` / `len`.
    pub fn output_name(&self) -> PlSmallStr {
        match self {
            Expr::Alias(_, name) => name.clone(),
            Expr::Column(name) => name.clone(),
            Expr::Wildcard => "*".into(),
            Expr::Literal(_) => "literal".into(),
            Expr::Len => "len".into(),
            Expr::Binary { left, right, .. } => match left.root_name() {
                Some(name) => name,
                None => right.output_name(),
            },
            Expr::Ternary { truthy, .. } => truthy.output_name(),
            Expr::Not(e) | Expr::Neg(e) | Expr::IsNull(e) | Expr::IsNotNull(e) => e.output_name(),
            Expr::Agg { input, .. } => input.output_name(),
            Expr::Cast { expr, .. } => expr.output_name(),
        }
    }

    fn root_name(&self) -> Option<PlSmallStr> {
        match self {
            Expr::Literal(_) => None,
            other => Some(other.output_name()),
        }
    }

    /// The columns this expression reads, appended to `out` (no duplicates).
    pub fn collect_columns(&self, out: &mut Vec<PlSmallStr>) {
        match self {
            Expr::Column(name) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Expr::Wildcard | Expr::Literal(_) | Expr::Len => {}
            Expr::Binary { left, right, .. } => {
                left.collect_columns(out);
                right.collect_columns(out);
            }
            Expr::Ternary { predicate, truthy, falsy } => {
                predicate.collect_columns(out);
                truthy.collect_columns(out);
                falsy.collect_columns(out);
            }
            Expr::Not(e) | Expr::Neg(e) | Expr::IsNull(e) | Expr::IsNotNull(e) | Expr::Alias(e, _) => e.collect_columns(out),
            Expr::Agg { input, .. } => input.collect_columns(out),
            Expr::Cast { expr, .. } => expr.collect_columns(out),
        }
    }

    /// `true` if the expression contains [`Expr::Wildcard`].
    pub fn has_wildcard(&self) -> bool {
        match self {
            Expr::Wildcard => true,
            Expr::Column(_) | Expr::Literal(_) | Expr::Len => false,
            Expr::Binary { left, right, .. } => left.has_wildcard() || right.has_wildcard(),
            Expr::Ternary { predicate, truthy, falsy } => {
                predicate.has_wildcard() || truthy.has_wildcard() || falsy.has_wildcard()
            }
            Expr::Not(e) | Expr::Neg(e) | Expr::IsNull(e) | Expr::IsNotNull(e) | Expr::Alias(e, _) => e.has_wildcard(),
            Expr::Agg { input, .. } => input.has_wildcard(),
            Expr::Cast { expr, .. } => expr.has_wildcard(),
        }
    }

    /// `true` if the expression reduces its input to a single value.
    pub fn is_aggregation(&self) -> bool {
        match self {
            Expr::Agg { .. } | Expr::Len => true,
            Expr::Alias(e, _) | Expr::Cast { expr: e, .. } | Expr::Not(e) | Expr::Neg(e) => e.is_aggregation(),
            Expr::Binary { left, right, .. } => {
                let scalar = |e: &Expr| e.is_aggregation() || matches!(e, Expr::Literal(_));
                scalar(left) && scalar(right) && (left.is_aggregation() || right.is_aggregation())
            }
            _ => false,
        }
    }
}

macro_rules! expr_arith {
    ($($trait:ident, $method:ident, $op:ident;)*) => {$(
        impl<R: Into<Expr>> $trait<R> for Expr {
            type Output = Expr;
            fn $method(self, rhs: R) -> Expr {
                self.binary(Operator::Arith(ArithOp::$op), rhs)
            }
        }
    )*};
}
expr_arith! { Add, add, Add; Sub, sub, Sub; Mul, mul, Mul; Div, div, Div; Rem, rem, Rem; }

impl Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self))
    }
}

impl Not for Expr {
    type Output = Expr;
    fn not(self) -> Expr {
        Expr::Not(Box::new(self))
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "col(\"{name}\")"),
            Expr::Wildcard => f.write_str("*"),
            Expr::Literal(LiteralValue::Null) => f.write_str("null"),
            Expr::Literal(LiteralValue::Boolean(b)) => write!(f, "{b}"),
            Expr::Literal(LiteralValue::Int(i)) => write!(f, "{i}"),
            Expr::Literal(LiteralValue::Float(x)) => write!(f, "{x:?}"),
            Expr::Literal(LiteralValue::String(s)) => write!(f, "\"{s}\""),
            Expr::Binary { left, op, right } => {
                let symbol = match op {
                    Operator::Cmp(CmpOp::Eq) => "==",
                    Operator::Cmp(CmpOp::Ne) => "!=",
                    Operator::Cmp(CmpOp::Lt) => "<",
                    Operator::Cmp(CmpOp::Le) => "<=",
                    Operator::Cmp(CmpOp::Gt) => ">",
                    Operator::Cmp(CmpOp::Ge) => ">=",
                    Operator::Arith(a) => a.symbol(),
                    Operator::And => "&",
                    Operator::Or => "|",
                    Operator::Xor => "^",
                };
                write!(f, "[({left}) {symbol} ({right})]")
            }
            Expr::Not(e) => write!(f, "{e}.not()"),
            Expr::Neg(e) => write!(f, "-({e})"),
            Expr::IsNull(e) => write!(f, "{e}.is_null()"),
            Expr::IsNotNull(e) => write!(f, "{e}.is_not_null()"),
            Expr::Agg { input, kind } => write!(f, "{input}.{}()", kind.suffix()),
            Expr::Len => f.write_str("len()"),
            Expr::Alias(e, name) => write!(f, "{e}.alias(\"{name}\")"),
            Expr::Cast { expr, dtype } => write!(f, "{expr}.cast({dtype})"),
            Expr::Ternary { predicate, truthy, falsy } => {
                write!(f, "when({predicate}).then({truthy}).otherwise({falsy})")
            }
        }
    }
}

/// Start a conditional expression: `when(cond).then(a).otherwise(b)`.
pub fn when(predicate: impl Into<Expr>) -> When {
    When { predicate: predicate.into() }
}

/// The `when(...)` part of a conditional expression.
#[derive(Clone, Debug)]
pub struct When {
    predicate: Expr,
}

/// The `when(...).then(...)` part of a conditional expression.
#[derive(Clone, Debug)]
pub struct Then {
    predicate: Expr,
    truthy: Expr,
}

impl When {
    /// The value where the condition holds.
    pub fn then(self, truthy: impl Into<Expr>) -> Then {
        Then { predicate: self.predicate, truthy: truthy.into() }
    }
}

impl Then {
    /// The value where the condition does not hold.
    pub fn otherwise(self, falsy: impl Into<Expr>) -> Expr {
        Expr::Ternary {
            predicate: Box::new(self.predicate),
            truthy: Box::new(self.truthy),
            falsy: Box::new(falsy.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builders_names_and_columns() {
        let e = (col("a") + 1).gt(col("b") * 2.5).and(col("c").is_null().not());
        let mut used = Vec::new();
        e.collect_columns(&mut used);
        assert_eq!(used, ["a", "b", "c"]);
        assert_eq!(e.output_name(), "a");
        assert_eq!(col("x").sum().alias("total").output_name(), "total");
        assert_eq!(lit(1).output_name(), "literal");
        assert_eq!((lit(1) + col("z")).output_name(), "z");
        assert_eq!(len().output_name(), "len");
        assert!(col("x").mean().is_aggregation());
        assert!((col("x").max() - col("x").min()).is_aggregation());
        assert!(!(col("x").max() - col("x")).is_aggregation());
        assert!(!col("x").is_aggregation());
        assert_eq!(col("*"), Expr::Wildcard);
        assert!(all().has_wildcard() && !col("a").has_wildcard());
        assert_eq!(cols(["a", "b"]).len(), 2);
        assert_eq!(
            col("a").is_between(1, 5, ClosedInterval::Left),
            col("a").gt_eq(1).and(col("a").lt(5))
        );
        assert_eq!(col("a").is_in([1, 2]), col("a").eq(1).or(col("a").eq(2)));
        assert_eq!(lit(None::<i32>), Expr::Literal(LiteralValue::Null));
        assert_eq!(
            col("a").lt(lit(5)).to_string(),
            "[(col(\"a\")) < (5)]"
        );
        let t = when(col("a").gt(0)).then(col("a")).otherwise(lit(0));
        assert_eq!(t.output_name(), "a");
    }
}
