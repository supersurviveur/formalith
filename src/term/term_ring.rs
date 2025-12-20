//! The group/ring/field containing expressions as constants.

use std::cmp::Ordering;

use crate::{
    field::{Field, Group, GroupBound, Ring, RingBound},
    printer::Print,
    term::flags::Flags,
};

use super::{Term, Value};

/// A ring where constants are complete expressions, like `2*x^2`. It is usefull for matrix and polynoms, to allow expressions inside coefficients.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TermField<T: GroupBound>(T);

impl<T: GroupBound> TermField<T> {
    /// Create a new term field for the set `T`
    pub const fn new(ring: T) -> Self {
        Self(ring)
    }
    /// Get the inner set
    pub fn get_set(&self) -> T {
        self.0
    }
}

impl<T: RingBound> Group for TermField<T> {
    type Element = Term<T>;

    type ExposantSet = Self;

    fn get_exposant_set(&self) -> Self::ExposantSet {
        *self
    }

    fn zero(&self) -> Self::Element {
        Term::Value(Value::new(self.get_set().zero(), self.get_set()))
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

    fn parse_litteral(&self, value: &str) -> Result<Term<T>, String> {
        Ok(Term::Value(Value::new(
            self.0.parse_litteral(value)?,
            self.get_set(),
        )))
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
        Print::pretty_print(elem, options)
    }
}

impl<T: RingBound> Ring for TermField<T> {
    fn one(&self) -> Self::Element {
        Term::Value(Value::new(self.get_set().one(), self.get_set()))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn nth(&self, nth: i64) -> Self::Element {
        Term::Value(Value::new(self.get_set().nth(nth), self.get_set()))
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        Some(a.inv())
    }
    fn normalize(&self, a: Term<Self>) -> Term<Self> {
        a.normalize()
    }
}

impl<T: Field> Field for TermField<T> {}
