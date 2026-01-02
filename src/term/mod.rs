//! Mathematical expression implementation. An expression is represented as a [Term], which can be of many types.
//! A term lives in a specific set, which can be a group, a ring, a field... Standard operations are defined between terms in [term_op].

use std::{
    borrow::BorrowMut,
    cmp::Ordering,
    fmt::{Debug, Display},
};

use crate::{
    context::{Context, Symbol},
    field::{
        Group, PartiallyOrderedSet, Ring, Set, TryElementFrom,
        matrix::{M, VectorSpaceElement},
    },
    matrix::MatrixResult,
    polynom::{MultivariatePolynomial, ToTerm},
    printer::{PrettyPrint, PrettyPrinter, Print, PrintOptions},
};

pub mod term_op;
pub mod term_ring;
pub use term_ring::*;
pub mod value;
pub use value::*;
pub mod symbol;
pub use symbol::*;
pub mod add;
pub use add::*;
pub mod mul;
pub use mul::*;
pub mod pow;
pub use pow::*;
pub mod fun;
pub use fun::*;
pub mod flags;
use flags::*;

/// A mathematical expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Term<T: Set> {
    /// See [Value]
    Value(Value<T>),
    /// See [SymbolTerm]
    Symbol(SymbolTerm<T>),
    /// See [Add]
    Add(Add<T>),
    /// See [Mul]
    Mul(Mul<T>),
    /// See [Pow]
    Pow(Pow<T>),
    /// See [Fun]
    Fun(Box<dyn Function<T>>),
}

impl<T: PartiallyOrderedSet> std::cmp::PartialOrd for Term<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Term::Value(Value { value: v1, .. }), Term::Value(Value { value: v2, .. })) => {
                self.get_set().partial_cmp(v1, v2)
            }
            (Term::Symbol(s1), Term::Symbol(s2)) => {
                if s1.symbol == s2.symbol {
                    Some(Ordering::Equal)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl<T: Group> Term<T> {
    /// Check if the expression is strictly positive
    pub fn is_strictly_positive(&self) -> bool {
        matches!(
            PartialOrd::partial_cmp(self, &Term::zero(self.get_set())),
            Some(Ordering::Greater)
        )
    }
    /// Check if the expression is positive
    pub fn is_positive(&self) -> bool {
        self.is_zero() || self.is_strictly_positive()
    }
    /// Check if the expression is strictly negative
    pub fn is_strictly_negative(&self) -> bool {
        matches!(
            PartialOrd::partial_cmp(self, &Term::zero(self.get_set())),
            Some(Ordering::Less)
        )
    }
    /// Check if the expression is negative
    pub fn is_negative(&self) -> bool {
        self.is_zero() || self.is_strictly_negative()
    }
}

impl<T: Set> Flags for Term<T> {
    fn get_flags(&self) -> u8 {
        match self {
            Term::Value(value) => value.get_flags(),
            Term::Symbol(symbol) => symbol.get_flags(),
            Term::Add(add) => add.get_flags(),
            Term::Mul(mul) => mul.get_flags(),
            Term::Pow(pow) => pow.get_flags(),
            Term::Fun(fun) => fun.get_flags(),
        }
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        match self {
            Term::Value(value) => value.get_flags_mut(),
            Term::Symbol(symbol) => symbol.get_flags_mut(),
            Term::Add(add) => add.get_flags_mut(),
            Term::Mul(mul) => mul.get_flags_mut(),
            Term::Pow(pow) => pow.get_flags_mut(),
            Term::Fun(fun) => fun.get_flags_mut(),
        }
    }
}

impl<T: Set> From<Mul<T>> for Term<T> {
    fn from(value: Mul<T>) -> Self {
        Term::Mul(value)
    }
}

impl<T: Set> From<Add<T>> for Term<T> {
    fn from(value: Add<T>) -> Self {
        Term::Add(value)
    }
}

impl<T: Set> From<Value<T>> for Term<T> {
    fn from(value: Value<T>) -> Self {
        Term::Value(value)
    }
}

impl<T: Set> From<Pow<T>> for Term<T> {
    fn from(value: Pow<T>) -> Self {
        Term::Pow(value)
    }
}

impl<T: Set> Term<T> {
    /// Absolute function
    pub const ABS: Symbol = Context::ABS;

    /// Get the ring where constants live
    pub fn get_set(&self) -> T {
        match self {
            Term::Value(value) => value.set,
            Term::Symbol(symbol) => symbol.set,
            Term::Add(add) => add.set,
            Term::Mul(mul) => mul.set,
            Term::Pow(pow) => pow.set,
            Term::Fun(fun) => fun.get_set(),
        }
    }
    /// Create a new empty product expression
    pub fn new_mul(ring: T) -> Self {
        Term::Mul(Mul::empty(ring))
    }
    /// Return `self` as [Value] if self is a `Value`, None otherwise.
    pub fn as_value(self) -> Option<Value<T>> {
        match self {
            Term::Value(value) => Some(value),
            _ => None,
        }
    }
    /// Return `self` as [Value].
    ///
    /// # Safety
    /// `self` must be a `Term::Value`, the method is UB otherwise.
    pub unsafe fn as_value_unsafe(self) -> Value<T> {
        match self {
            Term::Value(value) => value,
            _ => unreachable!(),
        }
    }
    /// Create a constant expression
    pub fn constant(constant: T::Element, set: T) -> Self {
        Term::Value(Value::new(constant, set))
    }
}
impl<T: Group> Term<T> {
    /// Create a zero constant expression
    pub fn zero(set: T) -> Self {
        Term::Value(Value::new(set.zero(), set))
    }
    /// Check if the term is zero
    /// ```
    /// use formalith::{field::R, parse, symbol};
    ///
    /// assert!(parse!("0", R).is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        match self {
            Term::Value(value) => value.set.is_zero(&value.value),
            Term::Symbol(_) | Term::Add(_) | Term::Mul(_) | Term::Pow(_) => false,
            Term::Fun(_) => false, // TODO how can we check if it's really not zero ?
        }
    }
}

impl<T: Group> Term<T> {
    /// Compare two terms
    #[allow(clippy::should_implement_trait)]
    pub fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Term::Value(Value { value: v1, .. }), Term::Value(Value { value: v2, .. })) => self
                .get_set()
                .partial_cmp(v1, v2)
                .unwrap_or(Ordering::Equal),
            (Term::Value(_), _) => Ordering::Greater,
            (_, Term::Value(_)) => Ordering::Less,
            (Term::Symbol(s1), Term::Symbol(s2)) => s1.symbol.cmp(&s2.symbol),
            (Term::Pow(p1), Term::Pow(p2)) => p1
                .base
                .cmp(&p2.base)
                .then_with(|| p1.exponant.cmp(&p2.exponant)),
            (Term::Pow(_), _) => Ordering::Less,
            (_, Term::Pow(_)) => Ordering::Greater,
            (Term::Mul(m1), Term::Mul(m2)) => {
                let len_cmp = m1.len().cmp(&m2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                for (a, b) in m1.iter().zip(m2.iter()) {
                    let cmp = a.cmp(b);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            }
            (Term::Mul(_), _) => Ordering::Less,
            (_, Term::Mul(_)) => Ordering::Greater,
            (Term::Add(a1), Term::Add(a2)) => {
                let len_cmp = a1.len().cmp(&a2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                for (a, b) in a1.iter().zip(a2.iter()) {
                    let cmp = a.cmp(b);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            }
            (Term::Add(_), _) => Ordering::Less,
            (_, Term::Add(_)) => Ordering::Greater,
            (Term::Symbol(_), _) => Ordering::Less,
            (_, Term::Symbol(_)) => Ordering::Greater,
            _ => todo!("Compare {:?} with {:?} is not implemented", self, other),
        }
    }
    /// Compare two terms, putting terms which can be merged side by side
    fn cmp_terms(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Term::Value(_), Term::Value(_)) => Ordering::Equal,
            (Term::Value(_), _) => Ordering::Greater,
            (_, Term::Value(_)) => Ordering::Less,
            (Term::Symbol(s1), Term::Symbol(s2)) => s1.symbol.cmp(&s2.symbol),
            (Term::Pow(p1), Term::Pow(p2)) => p1.base.cmp(&p2.base),
            (Term::Pow(p), _) => p.base.cmp(other).then(Ordering::Greater),
            (_, Term::Pow(p)) => self.cmp(&p.base).then(Ordering::Less),
            (Term::Mul(m1), Term::Mul(m2)) => {
                let len_cmp = m1.len().cmp(&m2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                // Compare non-coefficient part
                for (a, b) in m1.iter().zip(m2.iter()) {
                    // Skip coefficient
                    if let Term::Value(_) = a {
                        break;
                    }
                    if let Term::Value(_) = b {
                        break;
                    }

                    let non_coeff_cmp = a.cmp(b);
                    if non_coeff_cmp != Ordering::Equal {
                        return non_coeff_cmp;
                    }
                }
                Ordering::Equal
            }
            (Term::Mul(m1), v2) => {
                if m1.len() - m1.has_coeff() as usize != 1 {
                    Ordering::Greater
                } else {
                    // Compare the non-coefficient part
                    m1.factors[0].cmp(v2)
                }
            }
            (v1, Term::Mul(m2)) => {
                if m2.len() - m2.has_coeff() as usize != 1 {
                    Ordering::Less
                } else {
                    // Compare the non-coefficient part
                    v1.cmp(&m2.factors[0])
                }
            }
            (Term::Symbol(_), _) => Ordering::Less,
            (_, Term::Symbol(_)) => Ordering::Greater,
            _ => todo!(),
        }
    }
    /// Compare two terms, putting factors which can be merged side by side
    fn cmp_factors(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Term::Value(_), Term::Value(_)) => Ordering::Equal,
            (Term::Value(_), _) => Ordering::Greater,
            (_, Term::Value(_)) => Ordering::Less,
            (Term::Symbol(s1), Term::Symbol(s2)) => s1.symbol.cmp(&s2.symbol),
            (Term::Pow(p1), Term::Pow(p2)) => p1.base.cmp(&p2.base),
            (Term::Pow(p), _) => p.base.cmp(other).then(Ordering::Greater),
            (_, Term::Pow(p)) => self.cmp(&p.base).then(Ordering::Less),
            (Term::Add(m1), Term::Add(m2)) => {
                let len_cmp = m1.len().cmp(&m2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }
                // Compare all terms
                for (a, b) in m1.iter().zip(m2.iter()) {
                    let cmp = a.cmp(b);
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                Ordering::Equal
            }
            (Term::Symbol(_), _) => Ordering::Less,
            (_, Term::Symbol(_)) => Ordering::Greater,
            _ => todo!(),
        }
    }
}

impl<T: Ring> Term<M<TermSet<T>>> {
    /// Compute the determinant of the expression.
    pub fn det(&self) -> MatrixResult<Term<T>> {
        match self {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Scalar(scalar, _) => Ok(scalar.clone()),
                VectorSpaceElement::Vector(matrix) => Ok(matrix.det()?),
            },
            Term::Add(_) | Term::Mul(_) | Term::Pow(_) | Term::Symbol(_) | Term::Fun(_) => {
                Ok(Term::Fun(Box::new(Fun::new(
                    Context::DET,
                    vec![self.clone()],
                    *self.get_set().scalar_sub_set.get_set(),
                )) as Box<dyn Function<T>>))
            }
        }
    }
}

impl<T: Set> Term<T> {
    /// Convert `self` to [Mul], and return a mutable reference to easily edit it.
    fn convert_to_mul(&mut self) -> &mut Mul<T> {
        *self = Term::Mul(Mul::empty(self.get_set()));
        if let Term::Mul(n) = self {
            n
        } else {
            unreachable!()
        }
    }
    /// Convert `self` to [Pow], and return a mutable reference to easily edit it.
    fn convert_to_pow(
        &mut self,
        base: Box<Term<T>>,
        exponant: Box<Term<<T as Set>::ExponantSet>>,
    ) -> &mut Pow<T> {
        *self = Pow::new(base, exponant, self.get_set()).into();
        if let Term::Pow(n) = self {
            n
        } else {
            unreachable!()
        }
    }
}

trait MergeTerms {
    /// Try merging `other` term inside `self`. Return true if there was a merge.
    fn merge_terms(&mut self, other: &Self) -> bool;
}
trait MergeFactors {
    /// Try merging `other` factor inside `self`. Return true if there was a merge.
    fn merge_factors(&mut self, other: &Self) -> bool;
}
/// Normalization trait.
pub trait Normalize {
    /// Normalize an expression, i.e. sort sums and products, merge expressions that can be merged, remove useless expressions like zero in sums.
    fn normalize(&self) -> Self;
}
/// Unification trait.
pub trait Unify {
    /// Return the expression as a rational expression
    fn unify(&self) -> Self;
}

/// Expansion trait.
pub trait Expand {
    /// Expand the expression.
    /// ```
    /// use formalith::{field::R, parse, symbol};
    ///
    /// assert_eq!(parse!("(x+2)*(x+3)", R).expand(), parse!("x*x+5*x+6", R));
    /// ```
    fn expand(&self) -> Self;

    /// Expand the expression without normalizing it
    fn expand_without_norm(&self) -> Self;
}

impl<T: Set> MergeTerms for Term<T> {
    default fn merge_terms(&mut self, _other: &Self) -> bool {
        // There is no addition in a set
        false
    }
}

impl<T: Group> MergeTerms for Term<T> {
    default fn merge_terms(&mut self, other: &Self) -> bool {
        match (self.borrow_mut(), other) {
            (Term::Value(Value { value: v1, set, .. }), Term::Value(Value { value: v2, .. })) => {
                set.add_assign(v1, v2);
                true
            }
            (Term::Symbol(s1), Term::Symbol(s2)) if s1 == s2 => {
                // Merge x + x
                *self = Term::Mul(Mul::new(
                    s1.set.get_coefficient_set().nth(2),
                    vec![Term::Symbol(s1.clone())],
                    s1.set,
                ))
                .normalize();
                true
            }
            (Term::Mul(m1), Term::Mul(m2)) => {
                // Merge 2*x*y + 3*x*y
                let m1_len = m1.len() - m1.has_coeff() as usize;
                let m2_len = m2.len() - m2.has_coeff() as usize;
                if m1_len != m2_len {
                    return false;
                }
                // Compare non-coefficients factors
                for (a, b) in m1.iter().zip(m2.iter()) {
                    // Skip coefficients
                    if let Term::Value(_) = a {
                        break;
                    }
                    if let Term::Value(_) = b {
                        break;
                    }

                    if a != b {
                        return false;
                    }
                }

                m1.set_coeff(
                    m1.set
                        .get_coefficient_set()
                        .add(&m1.get_coeff(), &m2.get_coeff()),
                );
                *self = self.normalize();
                true
            }
            (Term::Mul(m1), _) => {
                // Merge 2*x + x
                if m1.len() - m1.has_coeff() as usize != 1 || m1.factors[0] != *other {
                    false
                } else {
                    let set = m1.set;
                    set.get_coefficient_set()
                        .add_assign(m1.get_coeff_mut(), &set.get_coefficient_set().nth(1));
                    true
                }
            }
            (_, Term::Mul(m2)) => {
                // Merge x + 2*x
                if m2.len() - m2.has_coeff() as usize != 1 || m2.factors[0] != *self {
                    false
                } else {
                    let set = m2.set;
                    let coeff = set
                        .get_coefficient_set()
                        .add(&m2.get_coeff(), &set.get_coefficient_set().nth(1));
                    if set.get_coefficient_set().is_zero(&coeff) {
                        *self = Value::new(set.zero(), set).into();
                    } else {
                        // Transform self to mul and add one to coeff
                        let mul = self.convert_to_mul();
                        mul.push(m2.factors[0].clone());
                        mul.set_coeff(coeff);
                    }
                    true
                }
            }
            (Term::Pow(p1), Term::Pow(p2)) if p1.base == p2.base && p1.exponant == p2.exponant => {
                // Merge x^2 + x^2
                *self = Term::Mul(Mul::new(
                    p1.set.get_coefficient_set().nth(2),
                    vec![other.clone()],
                    p1.set,
                ))
                .normalize();
                true
            }
            _ => false,
        }
    }
}

impl<T: Set> MergeFactors for Term<T> {
    default fn merge_factors(&mut self, _other: &Self) -> bool {
        // There is no product in a set and in a group
        false
    }
}

impl<T: Ring> Term<T> {
    fn merge_factors_in_ring(&mut self, other: &Self) -> bool {
        match (self.borrow_mut(), other) {
            (
                Term::Value(Value {
                    value: v1,
                    set: set1,
                    ..
                }),
                Term::Value(Value { value: v2, .. }),
            ) => {
                set1.mul_assign(v1, v2);
                return true;
            }
            (Term::Pow(p1), Term::Pow(p2)) => {
                // Merge x^2 * x^3
                if *p1.base != *p2.base {
                    return false;
                } else {
                    **p1.exponant = Term::Add(Add::new(
                        vec![(**p1.exponant).clone(), (**p2.exponant).clone()],
                        other.get_set().get_exponant_set(),
                    ));
                    **p1.exponant = (**p1.exponant).normalize();
                    return true;
                }
            }
            _ => {}
        }

        if self == other {
            // Merge x * x
            *self = Term::Pow(Pow::new(
                self.clone(),
                Term::Value(Value::new(
                    self.get_set().get_exponant_set().nth(2),
                    self.get_set().get_exponant_set(),
                )),
                self.get_set(),
            ))
            .normalize();
            return true;
        }
        false
    }
}

impl<T: Ring> MergeFactors for Term<T> {
    default fn merge_factors(&mut self, other: &Self) -> bool {
        self.merge_factors_in_ring(other)
    }
}
impl<T: Ring<ExponantSet: Ring>> MergeFactors for Term<T> {
    fn merge_factors(&mut self, other: &Self) -> bool {
        if !self.merge_factors_in_ring(other) {
            match (self.borrow_mut(), other) {
                (Term::Pow(p1), _) => {
                    // Merge x^2 * x
                    if *p1.base != *other {
                        false
                    } else if let Term::Value(Value { value, set, .. }) = &mut **p1.exponant {
                        // Add one to m1 coeff
                        set.add_assign(value, &set.one());
                        *self = self.normalize();
                        true
                    } else {
                        **p1.exponant = Term::Add(Add::new(
                            vec![
                                (**p1.exponant).clone(),
                                Value::new(
                                    other.get_set().get_exponant_set().one(),
                                    other.get_set().get_exponant_set(),
                                )
                                .into(),
                            ],
                            other.get_set().get_exponant_set(),
                        ));
                        **p1.exponant = (**p1.exponant).normalize();
                        true
                    }
                }
                (_, Term::Pow(p2)) => {
                    // Merge x * x^2
                    if *self != *p2.base {
                        false
                    } else if let Term::Value(Value { value, set, .. }) = &**p2.exponant {
                        // Transform self to mul and add one to coeff
                        self.convert_to_pow(
                            p2.base.clone(),
                            Term::Value(Value::new(set.add(value, &set.one()), *set)).into(),
                        );
                        *self = self.normalize();
                        true
                    } else {
                        // Add one to the coefficient
                        self.convert_to_pow(
                            p2.base.clone(),
                            Term::Add(Add::new(
                                vec![
                                    (**p2.exponant).clone(),
                                    Value::new(
                                        self.get_set().get_exponant_set().one(),
                                        self.get_set().get_exponant_set(),
                                    )
                                    .into(),
                                ],
                                self.get_set().get_exponant_set(),
                            ))
                            .into(),
                        );
                        *self = self.normalize();
                        true
                    }
                }
                _ => false,
            }
        } else {
            true
        }
    }
}

impl<T: Ring<ExponantSet: Ring>> Term<T> {
    /// Convert the expression to a multivariate polynomial.
    ///
    /// Used to use factorization algorithms.
    pub fn to_polynomial(
        &self,
    ) -> MultivariatePolynomial<Term<T>, TermSet<T>, TermSet<T::ExponantSet>> {
        self.expand().to_polynomial_impl()
    }

    fn to_polynomial_impl(
        &self,
    ) -> MultivariatePolynomial<Term<T>, TermSet<T>, TermSet<T::ExponantSet>> {
        match self {
            Term::Value(_) => MultivariatePolynomial::constant(
                self.clone(),
                self.get_set().get_term_set(),
                self.get_set().get_exponant_set().get_term_set(),
            ),
            Term::Symbol(symbol) => match Context::get_symbol_data(&symbol.symbol).name.as_str() {
                "n" => MultivariatePolynomial::constant(
                    self.clone(),
                    self.get_set().get_term_set(),
                    self.get_set().get_exponant_set().get_term_set(),
                ),
                _ => MultivariatePolynomial::variable(
                    self.clone(),
                    self.get_set().get_term_set(),
                    self.get_set().get_exponant_set().get_term_set(),
                ),
            },
            Term::Fun(_) => MultivariatePolynomial::variable(
                self.clone(),
                self.get_set().get_term_set(),
                self.get_set().get_exponant_set().get_term_set(),
            ),
            Term::Add(add) => {
                let mut res = MultivariatePolynomial::zero(
                    self.get_set().get_term_set(),
                    self.get_set().get_exponant_set().get_term_set(),
                );

                for term in add {
                    res = res + term.to_polynomial_impl()
                }

                res
            }
            Term::Mul(mul) => {
                let mut res = MultivariatePolynomial::one(
                    self.get_set().get_term_set(),
                    self.get_set().get_exponant_set().get_term_set(),
                );

                for factor in mul {
                    res = res * factor.to_polynomial_impl()
                }

                res
            }
            Term::Pow(pow) => MultivariatePolynomial::new(
                vec![(
                    vec![(*pow.base.clone(), (**pow.exponant).clone())],
                    self.get_set().get_term_set().one(),
                )],
                self.get_set().get_term_set(),
                self.get_set().get_exponant_set().get_term_set(),
            ),
        }
    }
}

impl<T: Ring> Expand for Term<T> {
    fn expand(&self) -> Self {
        debug_assert!(!self.needs_normalization());
        let res = self.expand_without_norm();
        let res = res.get_set().expand(res);
        res.normalize()
    }
    fn expand_without_norm(&self) -> Self {
        debug_assert!(!self.needs_normalization());
        match self {
            Term::Value(_) => self.clone(),
            Term::Symbol(_) => self.clone(),
            Term::Add(add) => Term::Add(Add::new(
                add.iter().map(|term| term.expand()).collect(),
                add.set,
            )),
            Term::Mul(mul) => {
                let mut mul = mul.clone();
                let (num, den) = self.as_fraction(false);
                if !den.is_one() {
                    let fraction = num.expand() / den.expand();
                    if let Term::Mul(fraction) = fraction {
                        mul = fraction;
                    } else {
                        return fraction;
                    };
                }
                let mut sums: Vec<Term<T>> = vec![];
                let mut new_sums = vec![];
                for factor in mul.iter() {
                    let expanded = factor.expand();
                    if let Term::Add(add) = expanded {
                        for term in add.iter() {
                            for sum in &sums {
                                if let Term::Mul(mul) = sum {
                                    let mut mul = mul.clone();
                                    mul.push(term.clone());
                                    new_sums.push(Term::Mul(mul));
                                } else {
                                    unreachable!()
                                }
                            }
                            if sums.is_empty() {
                                new_sums.push(Term::Mul(Mul::new(
                                    mul.coefficient.clone(),
                                    vec![term.clone()],
                                    mul.set,
                                )));
                            }
                        }
                        std::mem::swap(&mut sums, &mut new_sums);
                        new_sums.clear();
                    } else if sums.is_empty() {
                        sums.push(Term::Mul(Mul::new(
                            mul.coefficient.clone(),
                            vec![expanded],
                            mul.set,
                        )));
                    } else {
                        for sum in &mut sums {
                            if let Term::Mul(mul) = sum {
                                mul.push(expanded.clone());
                            } else {
                                unreachable!()
                            }
                        }
                    }
                }

                Term::Add(Add::new(sums, mul.set))
            }
            Term::Pow(pow) => {
                let mut res = pow.clone();
                *res.base = (*res.base).expand();
                **res.exponant = (**res.exponant).expand();

                Term::Pow(res)
            }
            Term::Fun(fun) => (**fun).expand(),
        }
    }
}

impl<T: Ring<ExponantSet: Ring>> Term<T> {
    /// Factor the expression.
    pub fn factor(&self) -> Self
    where
        TermSet<T>: TryElementFrom<TermSet<T::ExponantSet>>,
        Term<T>: TryFrom<Term<T::ExponantSet>>,
        <Term<T> as TryFrom<Term<T::ExponantSet>>>::Error: Debug,
    {
        let poly = self.to_polynomial();
        let factors = poly.factor();
        let mut res = Term::one(self.get_set());
        for (factor, multiplicity) in factors {
            res *= factor.to_term().pow(&Term::constant(
                self.get_set().get_exponant_set().nth(multiplicity.into()),
                self.get_set().get_exponant_set(),
            ))
        }
        res
    }
}
impl<T: Set> Expand for Term<T> {
    default fn expand(&self) -> Self {
        todo!()
    }
    default fn expand_without_norm(&self) -> Self {
        todo!()
    }
}
impl<T: Set> Normalize for Term<T> {
    default fn normalize(&self) -> Self {
        todo!()
    }
}
impl<T: Ring> Term<T> {
    fn normalize_in_ring(&self) -> Self {
        // return self.clone();
        // if !self.needs_normalization() {
        //     return self.clone();
        // }
        let mut res = match self {
            Term::Value(_) => self.clone(),
            Term::Symbol(_) => self.clone(),
            Term::Add(add) => {
                let mut res = vec![];
                for term in add.iter() {
                    let normalized = term.normalize();
                    if normalized.is_zero() {
                        continue;
                    }
                    if let Term::Add(inner_add) = normalized {
                        for term in inner_add.terms {
                            // terms are already normalized
                            res.push(term);
                        }
                    } else {
                        res.push(normalized);
                    }
                }
                if res.is_empty() {
                    return Term::zero(self.get_set());
                }
                res.sort_by(|a, b| a.cmp_terms(b));
                res.reverse();

                // Merge terms
                let mut terms = vec![];
                let mut current_merge = res.pop().unwrap();
                while let Some(next) = res.pop() {
                    if !current_merge.merge_terms(&next) {
                        terms.push(current_merge);
                        current_merge = next;
                    }
                }
                terms.push(current_merge);

                if terms.len() == 1 {
                    terms.pop().unwrap()
                } else {
                    Term::Add(Add::new(terms, self.get_set()))
                }
            }
            Term::Mul(mul) => {
                let mut res = vec![];
                let mut coeff = mul.coefficient.clone();
                for factor in mul.iter() {
                    let normalized = factor.normalize();
                    if normalized.is_zero() {
                        return Term::zero(self.get_set());
                    }
                    if normalized.is_one() {
                        continue;
                    }
                    if let Term::Mul(inner_mul) = normalized {
                        for factor in inner_mul.factors {
                            // factors are already normalized
                            res.push(factor);
                        }
                        mul.set
                            .get_coefficient_set()
                            .mul_assign(&mut coeff, &inner_mul.coefficient);
                    } else {
                        res.push(normalized);
                    }
                }
                if res.is_empty() {
                    if mul.set.get_coefficient_set().is_one(&coeff) {
                        return Term::one(self.get_set());
                    } else {
                        return Term::zero(self.get_set());
                    }
                }
                res.sort_by(Term::<T>::cmp_factors);
                res.reverse();

                // if res.len() == 2 {
                //     println!("{} {:?}", mul, res);
                // }

                // Merge factors
                let mut factors = vec![];
                let mut current_merge = res.pop().unwrap();

                while let Some(next) = res.pop() {
                    if !current_merge.merge_factors(&next) {
                        if current_merge.is_zero() {
                            return Term::zero(self.get_set());
                        }
                        factors.push(current_merge);
                        current_merge = next;
                    }
                }
                factors.push(current_merge);

                if mul.set.get_coefficient_set().is_zero(&coeff) {
                    Term::zero(self.get_set())
                } else if factors.len() == 1 && mul.set.get_coefficient_set().is_one(&coeff) {
                    factors.pop().unwrap()
                } else {
                    Term::Mul(Mul::new(coeff.clone(), factors, self.get_set()))
                }
            }
            Term::Pow(pow) => {
                let base = (*pow.base).normalize();
                let exponant = (**pow.exponant).normalize();
                if exponant.is_zero() {
                    return Term::one(self.get_set());
                }
                Term::Pow(Pow::new(Box::new(base), Box::new(exponant), self.get_set()))
            }
            Term::Fun(fun) => (**fun).normalize(),
        };
        res = res.get_set().normalize(res);
        res.set_normalized(true);
        res
    }
}
impl<T: Ring<ExponantSet: Ring>> Term<T> {
    fn normalize_in_ring_with_ring_exponant(&self) -> Self {
        let normalized = self.normalize_in_ring();
        if let Term::Pow(pow) = &normalized
            && pow.exponant.is_one()
        {
            return *pow.base.clone();
        }
        normalized
    }
}

impl<T: Ring> Normalize for Term<T> {
    default fn normalize(&self) -> Self {
        self.normalize_in_ring()
    }
}

impl<T: Ring<ExponantSet: Ring>> Normalize for Term<T> {
    default fn normalize(&self) -> Self {
        self.normalize_in_ring_with_ring_exponant()
    }
}
impl<T: Ring<ExponantSet: Ring, ProductCoefficientSet = T>> Normalize for Term<T> {
    fn normalize(&self) -> Self {
        if let Term::Mul(mut mul) = self.clone() {
            let mut new_coeff = mul.coefficient.clone();
            for factor in mul.iter() {
                if let Term::Value(Value { value, .. }) = factor {
                    mul.set
                        .get_coefficient_set()
                        .mul_assign(&mut new_coeff, value);
                }
            }
            mul.coefficient = new_coeff;
            mul.factors.retain(|elem| !matches!(elem, Term::Value(_)));
            if mul.is_empty() {
                Term::Value(Value::new(mul.coefficient, self.get_set()))
            } else {
                Term::Mul(mul).normalize_in_ring_with_ring_exponant()
            }
        } else {
            self.normalize_in_ring_with_ring_exponant()
        }
    }
}

impl<T: Ring> Term<T> {
    /// Return the expression as a rational expression, (numerator, denominator)
    pub fn as_fraction(&self, unify: bool) -> (Self, Self) {
        debug_assert!(!self.needs_normalization());
        match self {
            Term::Symbol(_) | Term::Fun(_) => (self.clone(), Term::one(self.get_set())),
            Term::Value(value) => {
                let (num, den) = self.get_set().as_fraction(&value.value);
                (
                    Term::constant(num, self.get_set()),
                    Term::constant(den, self.get_set()),
                )
            }
            Term::Mul(mul) => {
                let mut num = Term::one(self.get_set());
                let mut den = Term::one(self.get_set());

                for factor in mul {
                    let (n, d) = factor.as_fraction(unify);
                    num *= n;
                    den *= d;
                }
                let (n, d) = mul.set.get_coefficient_set().as_fraction(&mul.coefficient);
                if let Term::Mul(mul) = &mut num {
                    mul.set
                        .get_coefficient_set()
                        .mul_assign(&mut mul.coefficient, &n);
                } else {
                    num = Term::Mul(Mul::new(n, vec![num], mul.set)).normalize()
                }
                if let Term::Mul(mul) = &mut den {
                    mul.set
                        .get_coefficient_set()
                        .mul_assign(&mut mul.coefficient, &d);
                } else {
                    den = Term::Mul(Mul::new(d, vec![den], mul.set)).normalize()
                }
                (num, den)
            }
            Term::Add(add) => {
                let mut num = Term::zero(self.get_set());
                let mut den = Term::one(self.get_set());

                for term in add {
                    let (n, d) = term.as_fraction(unify);
                    if d == den {
                        num += n;
                    } else {
                        num *= &d;
                        num += n * &den;
                        den *= &d;
                    }
                }
                (num, den)
            }
            Term::Pow(pow) => {
                let mut pow = pow.clone();
                if unify {
                    *pow.base = pow.base.unify();
                    **pow.exponant = pow.exponant.unify();
                }
                if pow.exponant.is_strictly_negative() {
                    **pow.exponant = -&**pow.exponant;
                    (Term::one(self.get_set()), pow.into())
                } else {
                    (pow.into(), Term::one(self.get_set()))
                }
            }
        }
    }
}

impl<T: Set> Unify for Term<T> {
    default fn unify(&self) -> Self {
        self.clone()
    }
}

impl<T: Ring> Unify for Term<T> {
    fn unify(&self) -> Self {
        debug_assert!(!self.needs_normalization());
        let (num, den) = self.as_fraction(true);
        num / den
    }
}

impl<T: Ring> Term<T> {
    /// Simplify the expression by applying other simplification functions ([Self::unify], [Self::expand])
    pub fn simplify(&self) -> Self {
        debug_assert!(!self.needs_normalization());
        let mut res = self.get_set().simplify(self.clone());
        let (num, den) = res.as_fraction(true);
        res = num.expand() / den.expand(); // add set expansion if expand_without_norm and add a value_expand arg ?
        res = res.normalize();

        res
    }
}

impl<T: Ring> Term<T> {
    /// Invert the expression
    pub fn inv(&self) -> Self {
        debug_assert!(!self.needs_normalization());
        Term::one(self.get_set()) / self
    }
    /// Return the constant 1
    pub fn one(set: T) -> Self {
        Term::Value(Value::new(set.one(), set))
    }
    /// Check if the term is one
    /// ```
    /// use formalith::{field::R, parse, symbol};
    ///
    /// assert!(parse!("1", R).is_one());
    /// ```
    pub fn is_one(&self) -> bool {
        match self {
            Term::Value(value) => value.set.is_one(&value.value),
            Term::Symbol(_) | Term::Add(_) | Term::Mul(_) | Term::Pow(_) => false,
            Term::Fun(_) => todo!(),
        }
    }
}

// impl<T: Ring + Derivable> Term<T> {
//     pub fn derivative(&self, x: Symbol) -> Self {
//         match self {
//             Term::Value(value) => Term::Value(Value::new(
//                 value.ring.derivative(&value.value, x),
//                 value.ring,
//             )),
//             Term::Symbol(symbol) => {
//                 if symbol.symbol == x {
//                     Term::Value(Value::new(symbol.ring.one(), symbol.ring))
//                 } else {
//                     Term::Value(Value::new(symbol.ring.zero(), symbol.ring))
//                 }
//             }
//             Term::Add(add) => {
//                 let mut res = Add::with_capacity(add.len(), add.ring);
//                 for term in add.iter() {
//                     let derivated = term.derivative(x);
//                     if !derivated.is_zero() {
//                         res.push(term.derivative(x));
//                     }
//                 }
//                 Term::Add(res)
//             }
//             Term::Mul(mul) => {
//                 let mut res = Add::new(vec![], mul.ring);
//                 for i in 0..mul.len() {
//                     // derivate ith factor
//                     let mut tmp = Mul::with_capacity(mul.len(), mul.ring);
//                     let derivated = mul.factors[i].derivative(x);

//                     if derivated.is_zero() {
//                         continue;
//                     }

//                     for (j, factor) in mul.iter().enumerate() {
//                         if j != i {
//                             tmp.push(factor.clone());
//                         }
//                     }
//                     if !derivated.is_one() {
//                         tmp.push(derivated);
//                     }
//                     res.push(Term::Mul(tmp));
//                 }
//                 if res.len() == 0 {
//                     Term::Value(Value::new(mul.ring.zero(), mul.ring))
//                 } else if res.len() == 1 {
//                     res.terms.pop().unwrap()
//                 } else {
//                     Term::Add(res)
//                 }
//             }
//         }
//     }
// }

impl<T: Set> Display for Term<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        PrettyPrint::fmt(self, &PrintOptions::default(), f)
    }
}
impl<T: Set> Print for Term<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Value(value) => Print::print(value, options, f),
            Term::Symbol(symbol) => Print::print(symbol, options, f),
            Term::Add(add) => Print::print(add, options, f),
            Term::Mul(mul) => Print::print(mul, options, f),
            Term::Pow(pow) => Print::print(pow, options, f),
            Term::Fun(fun) => Print::print(&**fun, options, f),
        }
    }
}
impl<T: Set> PrettyPrint for Term<T> {
    fn pretty_print(&self, options: &PrintOptions) -> PrettyPrinter {
        match self {
            Term::Value(value) => PrettyPrint::pretty_print(value, options),
            Term::Symbol(symbol) => PrettyPrint::pretty_print(symbol, options),
            Term::Add(add) => PrettyPrint::pretty_print(add, options),
            Term::Mul(mul) => PrettyPrint::pretty_print(mul, options),
            Term::Pow(pow) => PrettyPrint::pretty_print(pow, options),
            Term::Fun(fun) => PrettyPrint::pretty_print(&**fun, options),
        }
    }
}
