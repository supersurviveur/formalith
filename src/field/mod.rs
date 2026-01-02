//! Sets specificities implementation.

use std::{cmp::Ordering, fmt, hash::Hash};

use crate::{
    context::Symbol,
    field::{integer::Z, matrix::M},
    parser::{Parser, ParserError},
    printer::{PrettyPrinter, PrintOptions},
    term::{Add, Mul, Pow, SymbolTerm, Term, TermSet, Value},
    traits::optional_default::OptionalDefault,
};

pub mod commons;
pub use commons::*;
use malachite::Integer;

/// A set.
pub trait Set: Clone + Copy + fmt::Debug + PartialEq + Eq + Hash + 'static {
    /// The type of the elements living in this set
    type Element: Clone + fmt::Debug + PartialEq + Eq + Hash;

    /// The set where exponants live. Mostly [const@integer::Z], but it can be any group,
    /// like [const@real::R] for real numbers since power is not only a notation for `x*x*...*x` but a defined operation.
    type ExponantSet: Group + OptionalDefault = Z<Integer>;

    /// The set where coefficients in product live. Mostly [const@integer::Z], but it can be any ring.
    type ProductCoefficientSet: Ring + OptionalDefault = Z<Integer>;

    /// Get the exponant set for this set
    fn get_exponant_set(&self) -> Self::ExponantSet;

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

    /// Get associated matrix set
    fn get_matrix_set(&self) -> M<Self> {
        M::new(*self)
    }

    /// Get associated term set
    fn get_term_set(&self) -> TermSet<Self> {
        TermSet::new(*self)
    }

    /// Compares two elements of this set and returns true if they are equals.
    fn element_eq(&self, a: &Self::Element, b: &Self::Element) -> bool;
}

/// A set with a partial ordering.
pub trait PartiallyOrderedSet: Set {
    /// Compares two elements of this set if possible, returns `None` otherwise
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering>;
}

/// A group is a set and a binary operation `+` and :
/// - `+` is associative
/// - `*` has an identity element `zero`
/// - Every element has an inverse element
pub trait Group: PartiallyOrderedSet {
    /// Get the zero (aka identity element for `+`) of this group
    fn zero(&self) -> Self::Element;
    /// Check if a number is zero.
    ///
    /// Defaults to an equality check with [Group::zero], but it's not always a desired behaviour :
    /// `10` in `Z/5Z` is zero (in its equivalence class)
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

/// [Set] trait with additional bounds, limiting exponant set recursion depth.
///
/// Auto implemented.
pub trait SetBound: Set<ExponantSet: Set<ExponantSet = Self::ExponantSet>> {}
impl<T: Set<ExponantSet: Set<ExponantSet = Self::ExponantSet>>> SetBound for T {}

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

impl<T: Set, From: Set> TryElementFrom<From> for T {
    default fn try_from_element(_value: From::Element) -> Result<Self::Element, TryCastError> {
        Err(TryCastError("Cast not implemented"))
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
    /// Defaults to an equality check with [Ring::one], but it's not always a desired behaviour :
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
    /// Return the expression unified
    fn unify(&self, a: &Self::Element) -> Self::Element {
        a.clone()
    }

    /// Normalize a mathematical expression using rules specific to this group. Check [real::R::normalize]
    fn normalize(&self, a: Term<Self>) -> Term<Self> {
        a
    }

    /// Expand a mathematical expression using rules specific to this group. Check [M::expand]
    fn expand(&self, a: Term<Self>) -> Term<Self> {
        a
    }

    /// Simplify a mathematical expression using rules specific to this group. Check [M::simplify]
    fn simplify(&self, a: Term<Self>) -> Term<Self> {
        a
    }
}

/// Parsing methods trait. The generic `N` represents the current recursion depth using the `typenum` crate.
pub trait SetParseExpression<N>: Set {
    /// Parses a string to an element of this set
    fn parse_literal<'a>(&self, parser: &mut Parser<'a>) -> Result<Option<Self::Element>, String>;

    /// Custom parsing function, to parse element specific to this set. Check [M::parse_expression] for example.
    fn parse_expression<'a>(
        &self,
        _parser: &mut Parser<'a>,
    ) -> Result<Option<Term<Self>>, ParserError> {
        Ok(None)
    }
}

/// Trait to cast an expression from a set to another set.
pub trait TryExprFrom<From: Set>: Set {
    /// Try to convert an expression over the set T into an expression over the set E.
    fn try_from_expr(&self, value: Term<From>) -> Result<Term<Self>, TryCastError>;
}
pub(crate) fn try_from_expr_default<From: Set, To: Set>(
    set: &To,
    value: Term<From>,
) -> Result<Term<To>, TryCastError> {
    match value {
        Term::Value(value) => Ok(Term::Value(Value::new(
            <To as TryElementFrom<From>>::try_from_element(value.value)?,
            *set,
        ))),
        Term::Symbol(symbol_term) => {
            Ok(Term::Symbol(SymbolTerm::new(symbol_term.symbol, *set)))
        }
        Term::Add(add) => Ok(Add::new(
            add.terms
                .into_iter()
                .map(|x| set.try_from_expr(x))
                .collect::<Result<Vec<_>, _>>()?,
            *set,
        )
        .into()),
        Term::Mul(mul) => {
            Ok(
                Mul::new(
                    <To::ProductCoefficientSet as TryElementFrom<
                        From::ProductCoefficientSet,
                    >>::try_from_element(mul.coefficient)?,
                    mul.factors
                        .into_iter()
                        .map(|x| TryExprFrom::<From>::try_from_expr(set, x))
                        .collect::<Result<Vec<_>, _>>()?,
                    *set,
                )
                .into(),
            )
        }
        Term::Pow(pow) => Ok(Pow::new(
            Box::new(TryExprFrom::<From>::try_from_expr(
                set,
                (*pow.base).clone(),
            )?),
            Box::new(TryExprFrom::<From::ExponantSet>::try_from_expr(
                &set.get_exponant_set(),
                (**pow.exponant).clone(),
            )?),
            *set,
        )
        .into()),
        Term::Fun(_) => {
            // Function can't be converted. Maybe with constraints we can check if some conversions are possible,
            // and then use a FunctionWrapper struct implementing Function trait to acheive the "conversion"
            Err(TryCastError("Can't cast function"))
        }
    }
}

impl<From: Set, To: Set> TryExprFrom<From> for To {
    default fn try_from_expr(&self, value: Term<From>) -> Result<Term<Self>, TryCastError> {
        try_from_expr_default(self, value)
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

/// Wrapper for an element living in a set. This wrapper implements all usefull traits like `Eq` or `Hash` using the implementation of the set itself.
/// It's usefull for sets like `${ZZ/p ZZ}$`, where equality is not the equality on the integer type.
#[derive(Eq)]
pub struct SetElement<T: Set>(T, T::Element);

impl<T: Set> Hash for SetElement<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
        self.1.hash(state);
    }
}

impl<T: Set> PartialEq for SetElement<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.0.element_eq(&self.1, &other.1)
    }
}

impl<T: Set> SetElement<T> {
    /// Wrap an element in a set.
    pub fn new(set: T, value: T::Element) -> Self {
        Self(set, value)
    }
}
