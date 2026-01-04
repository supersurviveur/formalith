//! Most commons groups, rings and fields like [struct@R] or [struct@M].
//! These are currently only implemented using malachite arbitrary precision numbers.

use malachite::rational::Rational;
use std::{
    error::Error,
    fmt::{self, Debug},
};

use crate::{
    field::{
        Set, TryExprFrom,
        matrix::{M, VectorSpaceElement},
        real::R,
        try_from_expr_default,
    },
    matrix::Matrix,
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
            VectorSpaceElement::Scalar(scalar, set) => Ok(Term::Value(Value::new(scalar, set))),
            VectorSpaceElement::Vector(_) => Err(TryCastError("Can't cast matrix to scalar")),
        }
    }
}

impl<From: Set, To: Set> TryExprFrom<TermSet<From>> for To {
    fn try_from_expr(&self, value: Term<TermSet<From>>) -> Result<Term<Self>, TryCastError> {
        match value {
            Term::Value(value) => {
                Ok(<TermSet<To> as TryElementFrom<TermSet<From>>>::try_from_element(value.value)?)
            }
            value => try_from_expr_default(self, value),
        }
    }
}

impl<T: Set> TryElementFrom<TermSet<T>> for T {
    fn try_from_element(value: Term<Self>) -> Result<Self::Element, TryCastError> {
        match value {
            Term::Value(value) => Ok(value.value),
            _ => Err(TryCastError(
                "Can't cast a non constant term to an element of T",
            )),
        }
    }
}

impl<T: Into<Rational>> From<T> for Term<R<Rational>> {
    fn from(value: T) -> Self {
        Self::Value(Value::new(value.into(), R))
    }
}

impl From<VectorSpaceElement<TermSet<R<Rational>>, Matrix<TermSet<R<Rational>>>>>
    for Term<M<TermSet<R<Rational>>>>
{
    fn from(value: VectorSpaceElement<TermSet<R<Rational>>, Matrix<TermSet<R<Rational>>>>) -> Self {
        Self::Value(Value::new(value, M))
    }
}
