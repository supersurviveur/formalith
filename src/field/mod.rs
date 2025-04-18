//! Sets specificities implementation.

use std::{cmp::Ordering, fmt};

use crate::{
    context::Symbol,
    parser::parser::{Parser, ParserError},
    term::Term,
};

pub mod commons;
pub use commons::*;

/// A group is a set and a binary operation `+` and :
/// - `+` is associative
/// - `*` has an identity element `zero`
/// - Every element has an inverse element
pub trait Group: Clone + fmt::Debug + PartialEq + 'static {
    /// The type of the elements living in this group
    type Element: Clone + fmt::Debug + fmt::Display + PartialEq + PartialOrd;

    /// The set where exposants live. Mostly [commons::Z], but it can be any ring,
    /// like [const@R] for real numbers since power is not only a notation for `x*x*...*x` but a defined operation.
    type ExposantSet: Ring;

    /// Get the exposant set for this group
    fn get_exposant_set(&self) -> &Self::ExposantSet;

    /// Get the zero (aka identity element for `+`) of this group
    fn zero(&self) -> Self::Element;
    /// Check if a number is zero.
    ///
    /// Defaults to an equality check with [Group::zero], but it's not always a desired behaviour :
    /// `10`in `Z/5Z` is zero (in its equivalence class)
    #[inline(always)]
    fn is_zero(&self, a: &Self::Element) -> bool {
        *a == self.zero()
    }
    /// Add two elements of this group.
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    /// Add two elements of this group into the first one.
    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(a, b);
    }
    /// Computes the symetric (inverse) element of `a` in this group for `+`
    fn neg(&self, a: &Self::Element) -> Self::Element;
    /// Computes the symetric (inverse) element of `a` in this group for `+` into `a`
    #[inline(always)]
    fn neg_assign(&self, a: &mut Self::Element) {
        *a = self.neg(a);
    }

    /// Compares two elements of this group if possible, returns `None` otherwise
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering>;

    /// Parses a string to an element of this group
    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String>;

    /// Custom parsing function, to parse element specific to this group. Check [commons::M::parse_expression] for example.
    fn parse_expression(
        &'static self,
        _parser: &mut Parser,
    ) -> Result<Option<Term<Self>>, ParserError> {
        Ok(None)
    }

    /// Normalize a mathematical expression using rules specific to this group. Check [commons::R::normalize]
    fn normalize(&'static self, a: Term<Self>) -> Term<Self> {
        a
    }
}

/// A ring is an abelian (commutative) [Group] with a binary operation `*` and :
/// - `*` is associative
/// - `*` has an identity element `one`
/// - `*` is distributive over `+`
pub trait Ring: Group {
    /// Get the one (aka identity element for `*`) of this group
    fn one(&self) -> Self::Element;
    /// Check if a number is one.
    ///
    /// Defaults to an equality check with [Ring::one], but it's not always a desired behaviour :
    /// `11`in `Z/5Z` is one (in its equivalence class)
    #[inline(always)]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == self.one()
    }
    /// Get the nth element by computing `n * 1`
    fn nth(&self, nth: i64) -> Self::Element;
    /// Multiply two elements of this ring
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    /// Multiply two elements of this ring into a
    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }
    /// Try computing the inverse element of a, returning `None` if it doesn't exist.
    fn inv(&self, a: &Self::Element) -> Option<Self::Element>;
}

/// A field is a [Ring] TODO
pub trait Field: Ring {}

/// TODO
pub trait Derivable: Ring {
    /// TODO rework derivative system
    fn derivative(&self, expr: &Self::Element, x: Symbol) -> Self::Element;
}
