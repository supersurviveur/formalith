//! The group/ring/field containing expressions as constants.

use std::cmp::Ordering;

use crate::field::{Field, Group, Ring};

use super::{Term, Value};

/// A ring where constants are complete expressions, like `2*x^2`. It is usefull for matrix and polynoms, to allow expressions inside coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct TermField<T: Group>(&'static T);

impl<T: Group> TermField<T> {
    /// Create a new term field for the set `T`
    pub const fn new(ring: &'static T) -> Self {
        Self(ring)
    }
    /// Get the inner set
    pub fn get_set(&self) -> &'static T {
        self.0
    }
}

impl<T: Ring> Group for TermField<T> {
    type Element = Term<T>;

    type ExposantSet = Self;

    fn get_exposant_set(&self) -> &Self::ExposantSet {
        todo!()
    }

    fn zero(&self) -> Self::Element {
        Term::Value(Value::new(self.get_set().zero(), self.get_set()))
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
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
}
impl<T: Ring> Ring for TermField<T> {
    fn one(&self) -> Self::Element {
        Term::Value(Value::new(self.get_set().one(), self.get_set()))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn nth(&self, nth: i64) -> Self::Element {
        Term::Value(Value::new(self.get_set().nth(nth), self.get_set()))
    }

    fn inv(&self, a: &Self::Element) -> Option<Self::Element> {
        Some(a.inv())
    }
}

impl<T: Field> Field for TermField<T> {}
