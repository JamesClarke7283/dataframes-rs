//! [`PlSmallStr`]: the string type used for column names.

use std::borrow::Borrow;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

const INLINE_CAP: usize = 22;

/// A cheaply cloneable immutable string, 24 bytes wide.
///
/// Names of up to 22 bytes are stored inline, so cloning a column — which
/// happens on every slice, select and filter — copies three words and touches
/// no reference count. Longer names share one heap allocation.
///
/// The name mirrors Polars' type so that `"a".into()` call sites port verbatim.
#[derive(Clone)]
pub struct PlSmallStr(Repr);

#[derive(Clone)]
enum Repr {
    Inline { len: u8, buf: [u8; INLINE_CAP] },
    Heap(Arc<str>),
}

impl PlSmallStr {
    /// The empty string.
    pub const EMPTY: PlSmallStr = PlSmallStr(Repr::Inline { len: 0, buf: [0; INLINE_CAP] });

    /// Build from a string slice (copies; inline when it fits).
    #[inline]
    pub fn from_str(s: &str) -> Self {
        if s.len() <= INLINE_CAP {
            let mut buf = [0u8; INLINE_CAP];
            buf[..s.len()].copy_from_slice(s.as_bytes());
            PlSmallStr(Repr::Inline { len: s.len() as u8, buf })
        } else {
            PlSmallStr(Repr::Heap(Arc::from(s)))
        }
    }

    /// Build from a `'static` string slice.
    #[inline]
    pub fn from_static(s: &'static str) -> Self {
        Self::from_str(s)
    }

    /// Build from an owned `String`.
    #[inline]
    pub fn from_string(s: String) -> Self {
        Self::from_str(&s)
    }

    /// The contents as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        match &self.0 {
            // SAFETY: the first `len` bytes were copied from a `&str` in
            // `from_str`, so they are valid UTF-8.
            Repr::Inline { len, buf } => unsafe { std::str::from_utf8_unchecked(&buf[..*len as usize]) },
            Repr::Heap(s) => s,
        }
    }

    /// Convert to an owned `String`.
    #[inline]
    pub fn into_string(self) -> String {
        self.as_str().to_string()
    }
}

impl Default for PlSmallStr {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl Deref for PlSmallStr {
    type Target = str;
    #[inline]
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl AsRef<str> for PlSmallStr {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl Borrow<str> for PlSmallStr {
    #[inline]
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl From<&str> for PlSmallStr {
    #[inline]
    fn from(s: &str) -> Self {
        Self::from_str(s)
    }
}

impl From<String> for PlSmallStr {
    #[inline]
    fn from(s: String) -> Self {
        Self::from_str(&s)
    }
}

impl From<&String> for PlSmallStr {
    #[inline]
    fn from(s: &String) -> Self {
        Self::from_str(s)
    }
}

impl From<&PlSmallStr> for PlSmallStr {
    #[inline]
    fn from(s: &PlSmallStr) -> Self {
        s.clone()
    }
}

impl From<PlSmallStr> for String {
    #[inline]
    fn from(s: PlSmallStr) -> Self {
        s.into_string()
    }
}

impl PartialEq for PlSmallStr {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}
impl Eq for PlSmallStr {}

impl PartialEq<str> for PlSmallStr {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        match &self.0 {
            // Column names are short: a length check and a byte loop beat a
            // call into `memcmp`, and most candidates fail on the length.
            Repr::Inline { len, buf } => {
                let bytes = other.as_bytes();
                if *len as usize != bytes.len() {
                    return false;
                }
                let mut i = 0;
                while i < bytes.len() {
                    if buf[i] != bytes[i] {
                        return false;
                    }
                    i += 1;
                }
                true
            }
            Repr::Heap(s) => &**s == other,
        }
    }
}

impl PartialEq<&str> for PlSmallStr {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        <PlSmallStr as PartialEq<str>>::eq(self, other)
    }
}

impl PartialEq<String> for PlSmallStr {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.as_str() == other.as_str()
    }
}

impl PartialEq<PlSmallStr> for &str {
    #[inline]
    fn eq(&self, other: &PlSmallStr) -> bool {
        *self == other.as_str()
    }
}

impl PartialOrd for PlSmallStr {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PlSmallStr {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl Hash for PlSmallStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl fmt::Display for PlSmallStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for PlSmallStr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self.as_str(), f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_and_heap_behave_alike() {
        assert_eq!(size_of::<PlSmallStr>(), 24);
        let short: PlSmallStr = "price".into();
        let long: PlSmallStr = "a column name that is longer than twenty-two bytes".into();
        assert!(matches!(short.0, Repr::Inline { .. }));
        assert!(matches!(long.0, Repr::Heap(_)));
        assert_eq!(short, "price");
        assert_eq!(long.len(), 50);
        assert_eq!(long.clone(), long);
        assert!(short < long.clone() || short > long);
        assert_eq!(PlSmallStr::EMPTY.as_str(), "");
        assert_eq!(format!("{short}/{short:?}"), "price/\"price\"");
        let exact: PlSmallStr = "x".repeat(22).into();
        assert!(matches!(exact.0, Repr::Inline { .. }));
        assert_eq!(exact.as_str(), "x".repeat(22));
        let unicode: PlSmallStr = "naïve—ß".into();
        assert_eq!(unicode.as_str(), "naïve—ß");
    }
}
