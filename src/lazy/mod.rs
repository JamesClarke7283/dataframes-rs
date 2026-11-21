//! Layer 6 — expressions and lazy queries.
//!
//! [`col`], [`lit`] and the methods on [`Expr`] build expressions;
//! [`LazyFrame`] strings them into a plan; `collect` runs it. Filters and
//! filtered aggregations are compiled to block-wise kernels and fused with
//! the compaction (see `exec`); everything else is evaluated column at a time.

pub(crate) mod dsl;
pub(crate) mod eval;
pub(crate) mod exec;
pub(crate) mod frame;
pub(crate) mod query;

pub use dsl::{ClosedInterval, Expr, Literal, LiteralValue, Operator, Then, When, all, col, cols, len, lit, when};
pub use frame::{IntoLazy, LazyFrame, LazyGroupBy};
pub use query::parse_query;
