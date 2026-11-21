//! Error type.
//!
//! The variants and the `PolarsError` / `PolarsResult` aliases mirror the names
//! used by Polars so that code written against `polars::prelude::*` ports by
//! changing a single `use` line.

use std::borrow::Cow;
use std::fmt;

/// Error message payload (`&'static str` or an owned `String`).
pub type ErrString = Cow<'static, str>;

/// Every fallible operation in this crate returns this error.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A column name was not present in the frame.
    ColumnNotFound(ErrString),
    /// A computation could not be carried out (overflow, parse failure, ...).
    ComputeError(ErrString),
    /// A column name occurs more than once.
    Duplicate(ErrString),
    /// The operation is not defined for the given data type.
    InvalidOperation(ErrString),
    /// An operation needed at least one value and got none.
    NoData(ErrString),
    /// A row or column index was past the end.
    OutOfBounds(ErrString),
    /// Data types do not line up.
    SchemaMismatch(ErrString),
    /// Lengths / heights do not line up.
    ShapeMismatch(ErrString),
    /// An I/O error, flattened to its message so the error stays `Clone`.
    IO(ErrString),
}

/// Polars-compatible alias for [`Error`].
pub type PolarsError = Error;
/// `Result` specialised to this crate's [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
/// Polars-compatible alias for [`Result`].
pub type PolarsResult<T> = Result<T>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (kind, msg) = match self {
            Error::ColumnNotFound(m) => ("column not found", m),
            Error::ComputeError(m) => ("compute error", m),
            Error::Duplicate(m) => ("duplicate", m),
            Error::InvalidOperation(m) => ("invalid operation", m),
            Error::NoData(m) => ("no data", m),
            Error::OutOfBounds(m) => ("out of bounds", m),
            Error::SchemaMismatch(m) => ("schema mismatch", m),
            Error::ShapeMismatch(m) => ("shape mismatch", m),
            Error::IO(m) => ("io error", m),
        };
        write!(f, "{kind}: {msg}")
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::IO(e.to_string().into())
    }
}

/// Build an [`Error`] of the given variant with a formatted message.
#[macro_export]
macro_rules! polars_err {
    ($variant:ident: $($arg:tt)+) => {
        $crate::error::Error::$variant(format!($($arg)+).into())
    };
}

/// Return early with an [`Error`] of the given variant.
#[macro_export]
macro_rules! polars_bail {
    ($variant:ident: $($arg:tt)+) => {
        return Err($crate::polars_err!($variant: $($arg)+))
    };
}

/// Return early with an [`Error`] unless the condition holds.
#[macro_export]
macro_rules! polars_ensure {
    ($cond:expr, $variant:ident: $($arg:tt)+) => {
        if !$cond {
            $crate::polars_bail!($variant: $($arg)+);
        }
    };
}
