//! Most commons groups, rings and fields like [struct@R] or [struct@M].
//! These are currently only implemented using malachite arbitrary precision numbers.

use malachite::rational::Rational;
use std::{
    error::Error,
    fmt::{self, Debug},
};

use crate::{
    field::{
        Set,
        matrix::{M, VectorSpaceElement},
        real::R,
    },
    term::{Term, TermSet, Value},
};

use super::TryElementFrom;

pub mod integer;
pub mod matrix;
pub mod real;

/// Failed to convert a term into another term.
#[derive(Debug, Clone, Copy)]
pub struct TryCastError(pub(crate) &'static str);
impl Error for TryCastError {}

impl fmt::Display for TryCastError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("Can't cast to this group: {}", self.0))
    }
}

impl<T: Set> TryElementFrom<M<T>> for TermSet<T> {
    fn try_from_element(value: <M<T> as Set>::Element) -> Result<Self::Element, TryCastError> {
        match value {
            VectorSpaceElement::Scalar(scalar, _) => Ok(scalar),
            VectorSpaceElement::Vector(_) => Err(TryCastError("Can't cast matrix to scalar")),
        }
    }
}

impl From<usize> for Term<R<Rational>> {
    fn from(value: usize) -> Self {
        Term::Value(Value::new(Rational::from(value), R))
    }
}

impl<T> TryElementFrom<TermSet<R<T>>> for R<T>
where
    Self: Set,
{
    fn try_from_element(
        value: <TermSet<R<T>> as Set>::Element,
    ) -> Result<Self::Element, TryCastError> {
        match value {
            Term::Value(value) => Ok(value.get_value()),
            _ => Err(TryCastError("Value is not a constant")),
        }
    }
}
