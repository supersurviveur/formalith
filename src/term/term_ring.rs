//! The group/ring/field containing expressions as constants.

use std::cmp::Ordering;

use malachite::Integer;

use crate::{
    field::{Field, Group, Ring, RingBound, Set, Z},
    printer::{PrettyPrint, Print},
    term::{Normalize, flags::Flags},
    traits::optional_default::OptionalDefault,
};

use super::{Term, Value};

/// A set where constants are complete expressions, like `2*x^2`. It is usefull for matrix and polynoms, to allow expressions inside coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TermSet<T>(T);

impl<T> TermSet<T> {
    /// Create a new term field for the set `T`
    pub const fn new(ring: T) -> Self {
        Self(ring)
    }

    /// Get the inner set
    pub fn get_set(&self) -> &T {
        &self.0
    }
}

impl<T: Set> Set for TermSet<T> {
    default type Element = Term<T>;

    default type ExponantSet = Z<Integer>;

    default type ProductCoefficientSet = Z<Integer>;

    default fn get_exposant_set(&self) -> Self::ExponantSet {
        Self::ExponantSet::optional_default().unwrap()
    }
    default fn get_coefficient_set(&self) -> Self::ProductCoefficientSet {
        Self::ProductCoefficientSet::optional_default().unwrap()
    }

    default fn print(
        &self,
        _elem: &Self::Element,
        _options: &crate::printer::PrintOptions,
        _f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        todo!()
    }

    default fn pretty_print(
        &self,
        _elem: &Self::Element,
        _options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        todo!()
    }
    default fn parse_literal(&self, _value: &str) -> Result<Self::Element, String> {
        todo!()
    }
}
impl<T: Ring> Set for TermSet<T> {
    type Element = Term<T>;

    type ExponantSet = Self;

    type ProductCoefficientSet = Self;

    fn get_exposant_set(&self) -> Self::ExponantSet {
        *self
    }
    fn get_coefficient_set(&self) -> Self::ProductCoefficientSet {
        *self
    }

    fn print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Print::print(elem, options, f)
    }

    fn pretty_print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        PrettyPrint::pretty_print(elem, options)
    }
    fn parse_literal(&self, value: &str) -> Result<Term<T>, String> {
        Ok(Term::Value(Value::new(
            self.0.parse_literal(value)?,
            *self.get_set(),
        )))
    }
}

impl<T: Ring> Group for TermSet<T> {
    fn zero(&self) -> Self::Element {
        Term::Value(Value::new(self.get_set().zero(), *self.get_set()))
    }

    fn nth(&self, nth: i64) -> Self::Element {
        Term::Value(Value::new(self.get_set().nth(nth), *self.get_set()))
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        debug_assert!(!(a + b).needs_normalization());
        a + b
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }

    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        PartialOrd::partial_cmp(a, b)
    }
}

impl<T: Ring> Ring for TermSet<T> {
    fn one(&self) -> Self::Element {
        Term::Value(Value::new(self.get_set().one(), *self.get_set()))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        Some(a.inv())
    }
    fn normalize(&self, a: Term<Self>) -> Term<Self> {
        a.normalize()
    }
}

impl<T: Field + RingBound> Field for TermSet<T> {}
