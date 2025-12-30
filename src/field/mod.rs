//! Sets specificities implementation.

use std::{cmp::Ordering, fmt, hash::Hash};

use crate::{
    context::Symbol,
    parser::{Parser, ParserError},
    printer::{PrettyPrinter, PrintOptions},
    term::{Add, Mul, Pow, SymbolTerm, Term, TermSet, Value},
    traits::optional_default::OptionalDefault,
};

pub mod commons;
pub use commons::*;
use malachite::Integer;
use typenum::{U0, U1, U2, U3, U4, U5, U6, U7, U8, U9, U10};

/// A set.
pub trait Set: Clone + Copy + fmt::Debug + PartialEq + Eq + Hash + 'static {
    /// The type of the elements living in this set
    type Element: Clone + fmt::Debug + fmt::Display + PartialEq + Eq + PartialOrd + Hash;

    /// The set where exposants live. Mostly [const@commons::Z], but it can be any group,
    /// like [const@R] for real numbers since power is not only a notation for `x*x*...*x` but a defined operation.
    type ExponantSet: Group + OptionalDefault = Z<Integer>;

    /// The set where coefficients in product live. Mostly [const@commons::Z], but it can be any ring.
    type ProductCoefficientSet: Ring + OptionalDefault = Z<Integer>;

    /// Get the exposant set for this set
    fn get_exposant_set(&self) -> Self::ExponantSet;

    /// Get the coefficient set for this set
    fn get_coefficient_set(&self) -> Self::ProductCoefficientSet;

    /// Print an element of this set.
    fn print(
        &self,
        elem: &Self::Element,
        options: &PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result;
    /// Pretty print an element of this set.
    fn pretty_print(&self, elem: &Self::Element, options: &PrintOptions) -> PrettyPrinter;

    /// Parses a string to an element of this set
    fn parse_literal(&self, value: &str) -> Result<Self::Element, String>;

    /// Get associated matrix set
    fn get_matrix_set(&self) -> M<Self> {
        M::new(*self)
    }

    /// Get associated term set
    fn get_term_set(&self) -> TermSet<Self> {
        TermSet::new(*self)
    }
}

/// A group is a set and a binary operation `+` and :
/// - `+` is associative
/// - `*` has an identity element `zero`
/// - Every element has an inverse element
pub trait Group: Set {
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
    /// Get the nth element by computing `n * 1`
    fn nth(&self, nth: i64) -> Self::Element;
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
    /// Substract two elements of this group.
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.add(a, &self.neg(b))
    }
    /// Substract two elements of this group into the first one.
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(a, b);
    }

    /// Compares two elements of this group if possible, returns `None` otherwise
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering>;

    /// Checks if a is greater or equal than zero in this group.
    fn is_positive(&self, a: &Self::Element) -> bool {
        self.is_zero(a) || self.is_strictly_positive(a)
    }

    /// Checks if a is greater than zero in this group.
    fn is_strictly_positive(&self, a: &Self::Element) -> bool {
        matches!(self.partial_cmp(a, &self.zero()), Some(Ordering::Greater))
    }

    /// Checks if a is less or equal than zero in this group.
    fn is_negative(&self, a: &Self::Element) -> bool {
        self.is_zero(a) || self.is_strictly_negative(a)
    }

    /// Checks if a is less than zero in this group.
    fn is_strictly_negative(&self, a: &Self::Element) -> bool {
        matches!(self.partial_cmp(a, &self.zero()), Some(Ordering::Less))
    }
}

/// [Set] trait with additional bounds, limiting exposant set recursion depth.
///
/// Auto implemented.
pub trait SetBound: Set<ExponantSet: Set<ExponantSet = Self::ExponantSet>> {}
impl<T: Set<ExponantSet: Set<ExponantSet = Self::ExponantSet>>> SetBound for T {}

/// [Group] trait with additional bounds, limiting exposant set recursion depth.
///
/// Auto implemented.
pub trait GroupBound: Group<ExponantSet: Group> + SetBound {}
impl<T: Group<ExponantSet: Group> + SetBound> GroupBound for T {}

/// Same as [GroupBound] for [Ring] trait.
///
/// Auto implemented.
pub trait RingBound: Ring<ExponantSet: Ring> {}
impl<T: Ring<ExponantSet: Ring>> RingBound for T {}

/// Trait to cast an element from a set to another set.
pub trait TryElementFrom<T: Set>: Set {
    /// Try to convert an element of the set T into an element of the set E.
    fn try_from_element(value: T::Element) -> Result<Self::Element, TryCastError>;
}

// Default implementation
impl<T: Set> TryElementFrom<T> for T {
    fn try_from_element(value: T::Element) -> Result<Self::Element, TryCastError> {
        Ok(value)
    }
}

/// A ring is an abelian (commutative) [Group] with a binary operation `*` and :
/// - `*` is associative
/// - `*` has an identity element `one`
/// - `*` is distributive over `+`
pub trait Ring: GroupBound {
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
    /// Multiply two elements of this ring
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    /// Multiply two elements of this ring into a
    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }
    /// Try computing the inverse element of a, returning `None` if it doesn't exist.
    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element>;

    /// Return the expression as a rational expression, (numerator, denominator)
    fn as_fraction(&self, a: &Self::Element) -> (Self::Element, Self::Element) {
        (a.clone(), self.one())
    }

    /// Normalize a mathematical expression using rules specific to this group. Check [commons::R::normalize]
    fn normalize(&self, a: Term<Self>) -> Term<Self> {
        a
    }

    /// Expand a mathematical expression using rules specific to this group. Check [commons::M::expand]
    fn expand(&self, a: Term<Self>) -> Term<Self> {
        a
    }

    /// Simplify a mathematical expression using rules specific to this group. Check [commons::M::simplify]
    fn simplify(&self, a: Term<Self>) -> Term<Self> {
        a
    }
}

/// Parsing methods trait. The generic `N` represents the current recursion depth using the `typenum` crate.
pub trait SetParseExpression<N>: Set {
    /// Custom parsing function, to parse element specific to this group. Check [commons::M::parse_expression] for example.
    fn parse_expression<'a>(
        &self,
        _parser: &mut Parser<'a>,
    ) -> Result<Option<Term<Self>>, ParserError> {
        Ok(None)
    }
}

/// [SetParseExpression] trait, enforcing `Self::ExponantSet` to also implements `SetParseExpression`
pub trait SetParseExpressionExponent<N>:
    SetParseExpression<N>
    + Set<
        ExponantSet: SetParseExpression<N> + Set<ProductCoefficientSet: SetParseExpression<N>>,
        ProductCoefficientSet: SetParseExpression<N>,
    >
{
}

impl<
    N,
    T: SetParseExpression<N>
        + Set<
            ExponantSet: SetParseExpression<N> + Set<ProductCoefficientSet: SetParseExpression<N>>,
            ProductCoefficientSet: SetParseExpression<N>,
        >,
> SetParseExpressionExponent<N> for T
{
}

/// [SetParseExpressionExponent] trait, enforcing `SetParseExpressionExponent<N>` where `N` is between 0 and 10.
pub trait SetParseExpressionBound:
    SetParseExpressionExponent<U0>
    + SetParseExpressionExponent<U1>
    + SetParseExpressionExponent<U2>
    + SetParseExpressionExponent<U3>
    + SetParseExpressionExponent<U4>
    + SetParseExpressionExponent<U5>
    + SetParseExpressionExponent<U6>
    + SetParseExpressionExponent<U7>
    + SetParseExpressionExponent<U8>
    + SetParseExpressionExponent<U9>
    + SetParseExpressionExponent<U10>
{
}

impl<T> SetParseExpressionBound for T where
    T: SetParseExpressionExponent<U0>
        + SetParseExpressionExponent<U1>
        + SetParseExpressionExponent<U2>
        + SetParseExpressionExponent<U3>
        + SetParseExpressionExponent<U4>
        + SetParseExpressionExponent<U5>
        + SetParseExpressionExponent<U6>
        + SetParseExpressionExponent<U7>
        + SetParseExpressionExponent<U8>
        + SetParseExpressionExponent<U9>
        + SetParseExpressionExponent<U10>
{
}

/// Trait to cast an expression from a set to another set.
pub trait TryExprFrom<From: Set>: Set {
    /// Try to convert an expression over the set T into an expression over the set E.
    fn try_from_expr(&self, value: Term<From>) -> Result<Term<Self>, TryCastError>;
}

impl<From: Set, To: SetBound + TryElementFrom<From>> TryExprFrom<From> for To
where
    To::ExponantSet: TryExprFrom<From::ExponantSet>,
    To::ProductCoefficientSet: TryElementFrom<From::ProductCoefficientSet>,
{
    fn try_from_expr(&self, value: Term<From>) -> Result<Term<Self>, TryCastError> {
        match value {
            Term::Value(value) => Ok(Term::Value(Value::new(
                <Self as TryElementFrom<From>>::try_from_element(value.value)?,
                *self,
            ))),
            Term::Symbol(symbol_term) => {
                Ok(Term::Symbol(SymbolTerm::new(symbol_term.symbol, *self)))
            }
            Term::Add(add) => Ok(Add::new(
                add.terms
                    .into_iter()
                    .map(|x| self.try_from_expr(x))
                    .collect::<Result<Vec<_>, _>>()?,
                *self,
            )
            .into()),
            Term::Mul(mul) => {
                Ok(
                    Mul::new(
                        <Self::ProductCoefficientSet as TryElementFrom<
                            From::ProductCoefficientSet,
                        >>::try_from_element(mul.coefficient)?,
                        mul.factors
                            .into_iter()
                            .map(|x| self.try_from_expr(x))
                            .collect::<Result<Vec<_>, _>>()?,
                        *self,
                    )
                    .into(),
                )
            }
            Term::Pow(pow) => Ok(Pow::new(
                Box::new(self.try_from_expr((*pow.base).clone())?),
                Box::new(
                    self.get_exposant_set()
                        .try_from_expr((**pow.exposant).clone())?,
                ),
                *self,
            )
            .into()),
            Term::Fun(_) => {
                // Function can't be converted. Maybe with constraints we can check if some conversions are possible,
                // and then use a FunctionWrapper struct implementing Function trait to acheive the "conversion"
                Err(TryCastError("Can't cast function"))
            }
        }
    }
}

/// An euclidean ring is a [Ring] TODO
pub trait EuclideanRing: Ring {
    /// Return the remainder of the euclidean division of a by b.
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    /// Compute the euclidean division of a by b, returning (quotient, remainder).
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element);
    /// Compute the greatest common divisor of a and b.
    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

/// A field is a [Ring] TODO
pub trait Field: Ring {
    /// Compute the inverse element of a.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        self.try_inv(a)
            .expect("Each elements should be inversible in a field !")
    }
}

/// TODO
pub trait Derivable: Ring {
    /// TODO rework derivative system
    fn derivative(&self, expr: &Self::Element, x: Symbol) -> Self::Element;
}
