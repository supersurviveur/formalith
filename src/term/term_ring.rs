use std::cmp::Ordering;

use crate::field::{Field, Group, Ring};

use super::{Term, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct TermField<T: Group>(&'static T);
impl<T: Group> TermField<T> {
    pub const fn new(ring: &'static T) -> Self {
        Self(ring)
    }
    pub fn get_ring(&self) -> &'static T {
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
        Term::Value(Value::new(self.get_ring().zero(), self.get_ring()))
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
        todo!()
    }
}
impl<T: Ring> Ring for TermField<T> {
    fn one(&self) -> Self::Element {
        Term::Value(Value::new(self.get_ring().one(), self.get_ring()))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn nth(&self, nth: i64) -> Self::Element {
        Term::Value(Value::new(self.get_ring().nth(nth), self.get_ring()))
    }

    fn inv(&self, a: &Self::Element) -> Option<Self::Element> {
        Some(a.inv())
    }
}

impl<T: Field> Field for TermField<T> {}
