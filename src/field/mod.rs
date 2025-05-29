//! Sets specificities implementation.

use std::{cmp::Ordering, fmt, hash::Hash};

use crate::{
    context::Symbol,
    parser::parser::{Parser, ParserError},
    printer::{PrettyPrinter, PrintOptions},
    term::{Add, Mul, Pow, SymbolTerm, Term, TermField, Value},
};

pub mod commons;
pub use commons::*;

/// A group is a set and a binary operation `+` and :
/// - `+` is associative
/// - `*` has an identity element `zero`
/// - Every element has an inverse element
pub trait GroupImpl: Clone + Copy + fmt::Debug + PartialEq + Eq + Hash + 'static {
    /// The type of the elements living in this group
    type Element: Clone + fmt::Debug + fmt::Display + PartialEq + Eq + PartialOrd + Hash;

    /// The set where exposants live. Mostly [commons::Z], but it can be any ring,
    /// like [const@R] for real numbers since power is not only a notation for `x*x*...*x` but a defined operation.
    type ExposantSet: GroupImpl;

    /// Get the exposant set for this group
    fn get_exposant_set(&self) -> Self::ExposantSet;

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

    /// Parses a string to an element of this group
    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String>;

    /// Pretty print an element of this set.
    fn pretty_print(&self, elem: &Self::Element, options: &PrintOptions) -> PrettyPrinter;
}

pub trait Group: GroupImpl<ExposantSet: GroupImpl<ExposantSet = Self::ExposantSet>> {}
impl<T: GroupImpl<ExposantSet: GroupImpl<ExposantSet = Self::ExposantSet>>> Group for T {}
pub trait Ring: RingImpl<ExposantSet: RingImpl<ExposantSet = Self::ExposantSet>> {}
impl<T: RingImpl<ExposantSet: RingImpl<ExposantSet = Self::ExposantSet>>> Ring for T {}

pub trait TryElementCast<T: GroupImpl>: GroupImpl {
    fn downcast_element(value: T::Element) -> Result<Self::Element, TryCastError>;
    fn upcast_element(value: Self::Element) -> Result<T::Element, TryCastError>;
}

impl<T: GroupImpl, U: GroupImpl> TryElementCast<T> for U {
    default fn downcast_element(
        value: <T as GroupImpl>::Element,
    ) -> Result<Self::Element, TryCastError> {
        todo!()
    }

    default fn upcast_element(
        value: Self::Element,
    ) -> Result<<T as GroupImpl>::Element, TryCastError> {
        todo!()
    }
}

/// A ring is an abelian (commutative) [Group] with a binary operation `*` and :
/// - `*` is associative
/// - `*` has an identity element `one`
/// - `*` is distributive over `+`
pub trait RingImpl: Group {
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
    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element>;

    /// Return the expression as a rational expression, (numerator, denominator)
    fn as_fraction(&self, a: &Self::Element) -> (Self::Element, Self::Element) {
        (a.clone(), self.one())
    }

    ///Custom parsing function, to parse element specific to this group. Check [commons::M::parse_expression] for example.
    fn parse_expression(&self, _parser: &mut Parser) -> Result<Option<Term<Self>>, ParserError>
    where
        Self: Group,
        Self::ExposantSet: Ring,
    {
        Ok(None)
    }

    /// Normalize a mathematical expression using rules specific to this group. Check [commons::R::normalize]
    fn normalize(&self, a: Term<Self>) -> Term<Self>
    where
        Self: Group,
        Self::ExposantSet: Ring,
    {
        a
    }

    /// Expand a mathematical expression using rules specific to this group. Check [commons::M::expand]
    fn expand(&self, a: Term<Self>) -> Term<Self>
    where
        Self: Group,
        Self::ExposantSet: Ring,
    {
        a
    }

    /// Simplify a mathematical expression using rules specific to this group. Check [commons::M::simplify]
    fn simplify(&self, a: Term<Self>) -> Term<Self>
    where
        Self: Group,
        Self::ExposantSet: Ring,
    {
        a
    }

    /// Get associated term field
    fn get_term_field(&self) -> TermField<Self>
    where
        Self: Group,
        Self::ExposantSet: Ring,
    {
        TermField::new(*self)
    }

    /// Get associated matrix ring
    fn get_matrix_ring(&self) -> M<Self>
    where
        Self: Group,
        Self::ExposantSet: Ring,
    {
        M::new(*self)
    }
}

/// Trait to downcast an expression from a set to another set.
///
/// It use horrible generic types to work, to avoid issues with infinite type with exposant set.
/// The hack is to allow only a fixed amount of exposant set (currently 3) that are not their own exposant set (for instance exposant set of R is R).
/// That way there is only a finite number of sets that can be converted.
/// This limit can be enlarged by adding more templating, but most set have an exposant set which is already cycling.
pub trait TryExprCast<From: Ring>: Ring {
    /// Try to convert an expression over the set T into an expression over the set E.
    ///
    /// See [Downcast] to understand the templating.
    fn downcast_expr(&self, value: Term<From>) -> Result<Term<Self>, TryCastError>;
    fn upcast_expr(set: From, value: Term<Self>) -> Result<Term<From>, TryCastError>;
}

impl<From: Ring, To: Ring> TryExprCast<From> for To
where
    Self: TryElementCast<From>,
    Self::ExposantSet: TryElementCast<From::ExposantSet>,
    <Self::ExposantSet as GroupImpl>::ExposantSet:
        TryElementCast<<From::ExposantSet as GroupImpl>::ExposantSet>,
    <Self::ExposantSet as GroupImpl>::ExposantSet:
        Group<ExposantSet = <Self::ExposantSet as GroupImpl>::ExposantSet>,
    <From::ExposantSet as GroupImpl>::ExposantSet:
        Group<ExposantSet = <From::ExposantSet as GroupImpl>::ExposantSet>,
{
    fn downcast_expr(&self, value: Term<From>) -> Result<Term<Self>, TryCastError> {
        match value {
            Term::Value(value) => Ok(Term::Value(Value::new(
                <Self as TryElementCast<From>>::downcast_element(value.value)
                    .map_err(|_| TryCastError())?,
                *self,
            ))),
            Term::Symbol(symbol_term) => {
                Ok(Term::Symbol(SymbolTerm::new(symbol_term.symbol, *self)))
            }
            Term::Add(add) => Ok(Add::new(
                add.terms
                    .into_iter()
                    .map(|x| self.downcast_expr(x))
                    .collect::<Result<Vec<_>, _>>()?,
                *self,
            )
            .into()),
            Term::Mul(mul) => Ok(Mul::new(
                mul.factors
                    .into_iter()
                    .map(|x| self.downcast_expr(x))
                    .collect::<Result<Vec<_>, _>>()?,
                *self,
            )
            .into()),
            Term::Pow(pow) => Ok(Pow::new(
                Box::new(self.downcast_expr((*pow.base).clone())?),
                Box::new(
                    self.get_exposant_set()
                        .downcast_expr((**pow.exposant).clone())?,
                ),
                *self,
            )
            .into()),
            Term::Fun(_) => {
                // Function can't be converted. Maybe with constraints we can check if some conversions are possible,
                // and then use a FunctionWrapper struct implementing Function trait to acheive the "conversion"
                Err(TryCastError())
            }
        }
    }

    fn upcast_expr(set: From, value: Term<Self>) -> Result<Term<From>, TryCastError> {
        match value {
            Term::Value(value) => Ok(Term::Value(Value::new(
                <Self as TryElementCast<From>>::upcast_element(value.value)
                    .map_err(|_| TryCastError())?,
                set,
            ))),
            Term::Symbol(symbol_term) => Ok(Term::Symbol(SymbolTerm::new(symbol_term.symbol, set))),
            Term::Add(add) => Ok(Add::new(
                add.terms
                    .into_iter()
                    .map(|x| Self::upcast_expr(set, x))
                    .collect::<Result<Vec<_>, _>>()?,
                set,
            )
            .into()),
            Term::Mul(mul) => Ok(Mul::new(
                mul.factors
                    .into_iter()
                    .map(|x| Self::upcast_expr(set, x))
                    .collect::<Result<Vec<_>, _>>()?,
                set,
            )
            .into()),
            Term::Pow(pow) => Ok(Pow::new(
                Box::new(Self::upcast_expr(set, (*pow.base).clone())?),
                Box::new(Self::ExposantSet::upcast_expr(
                    set.get_exposant_set(),
                    (**pow.exposant).clone(),
                )?),
                set,
            )
            .into()),
            Term::Fun(_) => Err(TryCastError()),
        }
    }
}

/// An euclidean ring is a [Ring] TODO
pub trait EuclideanRing: Ring {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element);
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
