//! The [OptionalDefault] trait for types with an optional default value.

use malachite::{Integer, Rational};

use crate::{
    field::{M, R, Z},
    term::TermSet,
};

/// Same as [Default], but if no default value is meaningfull, `None` can be returned.
pub trait OptionalDefault: Sized {
    /// Same as [Default::default], but optional.
    fn optional_default() -> Option<Self>;
}

impl OptionalDefault for Z<Integer> {
    fn optional_default() -> Option<Self> {
        Some(Z)
    }
}
impl OptionalDefault for R<Rational> {
    fn optional_default() -> Option<Self> {
        Some(R)
    }
}
impl<T> OptionalDefault for M<T> {
    fn optional_default() -> Option<Self> {
        None
    }
}
impl<T> OptionalDefault for TermSet<T> {
    fn optional_default() -> Option<Self> {
        None
    }
}
