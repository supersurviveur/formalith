//! The group/ring/field containing expressions as constants.

use std::cmp::Ordering;

use crate::{
    field::{Field, Group, PartiallyOrderedSet, Ring, Set, SetParseExpression, TryExprFrom},
    parser::{Parser, ParserError, ParserTraitBounded},
    printer::{PrettyPrint, Print},
    term::{Expand, Normalize, Unify, flags::Flags},
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
    type Element = Term<T>;

    type ExponantSet = TermSet<T::ExponantSet>;

    type ProductCoefficientSet = TermSet<T::ProductCoefficientSet>;

    fn get_exponant_set(&self) -> Self::ExponantSet {
        self.get_set().get_exponant_set().get_term_set()
    }

    fn get_coefficient_set(&self) -> Self::ProductCoefficientSet {
        self.get_set().get_coefficient_set().get_term_set()
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

    fn element_eq(&self, a: &Self::Element, b: &Self::Element) -> bool {
        a == b
    }
}

impl<T: PartiallyOrderedSet> PartiallyOrderedSet for TermSet<T> {
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        PartialOrd::partial_cmp(a, b)
    }
}

impl<T: Group> Group for TermSet<T> {
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
        let a = self.get_set().try_from_expr(a).unwrap();
        Term::Value(Value::new(a.normalize(), *self))
    }
    fn simplify(&self, a: Term<Self>) -> Term<Self> {
        let a = self.get_set().try_from_expr(a).unwrap();
        Term::Value(Value::new(a.simplify(), *self))
    }
    fn expand(&self, a: Term<Self>) -> Term<Self> {
        let a = self.get_set().try_from_expr(a).unwrap();
        Term::Value(Value::new(a.expand(), *self))
    }
    fn as_fraction(&self, a: &Self::Element) -> (Self::Element, Self::Element) {
        a.as_fraction(false)
    }
    fn unify(&self, a: &Self::Element) -> Self::Element {
        a.unify()
    }
}

impl<T: Field> Field for TermSet<T> {}
impl<T: Set, N> SetParseExpression<N> for TermSet<T> {
    fn parse_expression(&self, parser: &mut Parser) -> Result<Option<Term<Self>>, ParserError> {
        Ok(
            ParserTraitBounded::<T, N>::parse_expression_bounded(parser, 0, *self.get_set())?
                .map(|parsed| Term::Value(Value::new(parsed, *self))),
        )
    }

    fn parse_literal(&self, parser: &mut Parser<'_>) -> Result<Option<Self::Element>, String> {
        Ok(ParserTraitBounded::<T, N>::parse_expression_bounded(
            parser,
            0,
            *self.get_set(),
        )?)
    }
}
