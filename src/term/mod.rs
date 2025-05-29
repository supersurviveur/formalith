//! Mathematical expression implementation. An expression is represented as a [Term], which can be of many types.
//! A term lives in a specific set, which can be a group, a ring, a field... Standard operations are defined between terms in [term_op].

use std::{
    borrow::BorrowMut,
    cmp::Ordering,
    fmt::{Debug, Display},
};

use crate::{
    context::{Context, Symbol},
    field::{Group, GroupImpl, Ring, RingImpl, VectorSpaceElement, M},
    matrix::{Matrix, MatrixResult},
    polynom::MultivariatePolynomial,
    printer::{PrettyPrinter, Print, PrintOptions},
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
pub enum Term<T: Group> {
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

impl<T: Ring> std::cmp::PartialOrd for Term<T> {
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

impl<T: Ring> Term<T> {
    /// Check if the expression is strictly positive
    pub fn is_strictly_positive(&self) -> bool {
        match PartialOrd::partial_cmp(self, &Term::zero(self.get_set())) {
            Some(Ordering::Greater) => true,
            _ => false,
        }
    }
    /// Check if the expression is positive
    pub fn is_positive(&self) -> bool {
        self.is_zero() || self.is_strictly_positive()
    }
    /// Check if the expression is strictly negative
    pub fn is_strictly_negative(&self) -> bool {
        match PartialOrd::partial_cmp(self, &Term::zero(self.get_set())) {
            Some(Ordering::Less) => true,
            _ => false,
        }
    }
    /// Check if the expression is negative
    pub fn is_negative(&self) -> bool {
        self.is_zero() || self.is_strictly_negative()
    }
}

impl<T: Ring> Flags for Term<T> {
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

impl<T: Ring> From<Mul<T>> for Term<T> {
    fn from(value: Mul<T>) -> Self {
        Term::Mul(value)
    }
}

impl<T: Ring> From<Add<T>> for Term<T> {
    fn from(value: Add<T>) -> Self {
        Term::Add(value)
    }
}

impl<T: Ring> From<Value<T>> for Term<T> {
    fn from(value: Value<T>) -> Self {
        Term::Value(value)
    }
}

impl<T: Ring> From<Pow<T>> for Term<T> {
    fn from(value: Pow<T>) -> Self {
        Term::Pow(value)
    }
}

impl<T: Ring> Term<T> {
    /// Absolute function
    pub const ABS: Symbol = Context::ABS;

    /// Create a new empty product expression
    pub fn new_mul(ring: T) -> Self {
        Term::Mul(Mul::new(vec![], ring))
    }
    /// Get the ring where constants live
    pub fn get_set(&self) -> T {
        match self {
            Term::Value(value) => value.ring,
            Term::Symbol(symbol) => symbol.ring,
            Term::Add(add) => add.ring,
            Term::Mul(mul) => mul.ring,
            Term::Pow(pow) => pow.set,
            Term::Fun(fun) => fun.get_set(),
        }
    }
    /// Create a zero constant expression
    pub fn zero(set: T) -> Self {
        Term::Value(Value::new(set.zero(), set))
    }
    /// Create a constant expression
    pub fn constant(constant: T::Element, set: T) -> Self {
        Term::Value(Value::new(constant, set))
    }
    /// Check if the term is zero
    /// ```
    /// use formalith::{field::R, parse, symbol};
    ///
    /// assert!(parse!("0", R).is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        match self {
            Term::Value(value) => value.ring.is_zero(&value.value),
            Term::Symbol(_) | Term::Add(_) | Term::Mul(_) | Term::Pow(_) => false,
            Term::Fun(_) => false, // TODO how can we check if it's really not zero ?
        }
    }

    /// Return `self` as [Value]. It's UB if `self` is not a value.
    unsafe fn as_value(&mut self) -> &mut Value<T> {
        match self {
            Term::Value(value) => value,
            _ => unreachable!(),
        }
    }
}

impl<T: Ring> Term<T> {
    /// Compare two terms
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
                .then_with(|| p1.exposant.cmp(&p2.exposant)),
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
                if m1.len() != 2 {
                    Ordering::Greater
                } else {
                    // Compare the non-coefficient part
                    m1.factors[0].cmp(v2)
                }
            }
            (v1, Term::Mul(m2)) => {
                if m2.len() != 2 {
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

impl<T: Ring> Term<M<T>>
where
    M<T>: Ring<Element = VectorSpaceElement<T, Matrix<TermField<T>>>>,
{
    pub fn det(&self) -> MatrixResult<Term<T>> {
        match self {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Scalar(scalar) => Ok(Term::Value(Value::new(
                    scalar.clone(),
                    self.get_set().scalar_sub_set,
                ))),
                VectorSpaceElement::Vector(matrix) => Ok(matrix.det()?),
            },
            Term::Add(add) => todo!(),
            Term::Mul(mul) => todo!(),
            Term::Pow(pow) => todo!(),
            Term::Symbol(_) | Term::Fun(_) => Ok(Term::Fun(Box::new(Fun::new(
                Context::DET,
                vec![self.clone()],
                self.get_set().scalar_sub_set,
            )))),
        }
    }
}

impl<T: Ring> Term<T> {
    /// Try merging `other` term inside `self`. Return true if there was a merge.
    fn merge_terms(&mut self, other: &Self) -> bool {
        match (self.borrow_mut(), other) {
            (
                Term::Value(Value {
                    value: v1,
                    ring: ring1,
                    ..
                }),
                Term::Value(Value { value: v2, .. }),
            ) => {
                ring1.add_assign(v1, v2);
                true
            }
            (Term::Symbol(s1), Term::Symbol(s2)) if s1 == s2 => {
                // Merge x + x
                *self = Term::Mul(Mul::new(
                    vec![
                        Term::Symbol(s1.clone()),
                        Term::Value(Value::new(
                            s1.ring.add(&s1.ring.one(), &s1.ring.one()), // Use a integer to T conversion in Ring Trait instead ?
                            s1.ring,
                        )),
                    ],
                    s1.ring,
                ));
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

                m1.set_coeff(m1.ring.add(&m1.get_coeff(), &m2.get_coeff()));
                true
            }
            (Term::Mul(m1), _) => {
                // Merge 2*x + x
                if m1.len() != 2 || m1.factors[0] != *other {
                    false
                } else if let Term::Value(Value { value, ring, .. }) = &mut m1.factors[1] {
                    // Add one to m1 coeff
                    ring.add_assign(value, &ring.one());
                    true
                } else {
                    false
                }
            }
            (_, Term::Mul(m2)) => {
                // Merge x + 2*x
                if m2.len() != 2 || m2.factors[0] != *self {
                    false
                } else if let Term::Value(Value { value, ring, .. }) = &m2.factors[1] {
                    // Transform self to mul and add one to coeff
                    let mul = self.to_mul();
                    mul.push(m2.factors[0].clone());
                    mul.set_coeff(ring.add(value, &ring.one()));
                    true
                } else {
                    false
                }
            }
            (Term::Pow(p1), Term::Pow(p2)) if p1.base == p2.base && p1.exposant == p2.exposant => {
                // Merge x^2 + x^2
                *self = Term::Mul(Mul::new(
                    vec![
                        other.clone(),
                        Term::Value(Value::new(
                            p1.set.add(&p1.set.one(), &p1.set.one()), // Use a integer to T conversion in Ring Trait instead ?
                            p1.set,
                        )),
                    ],
                    p1.set,
                ));
                true
            }
            _ => false,
        }
    }

    /// Try merging `other` factor inside `self`. Return true if there was a merge.
    fn merge_factors(&mut self, other: &Self) -> bool {
        match (self.borrow_mut(), other) {
            (
                Term::Value(Value {
                    value: v1,
                    ring: ring1,
                    ..
                }),
                Term::Value(Value { value: v2, .. }),
            ) => {
                ring1.mul_assign(v1, v2);
                return true;
            }
            (Term::Pow(p1), Term::Pow(p2)) => {
                // Merge x^2 * x^3
                if *p1.base != *p2.base {
                    return false;
                } else {
                    **p1.exposant = Term::Add(Add::new(
                        vec![(**p1.exposant).clone(), (**p2.exposant).clone()],
                        other.get_set().get_exposant_set(),
                    ))
                    .into();
                    **p1.exposant = (**p1.exposant).normalize();
                    return true;
                }
            }
            (Term::Pow(p1), _) => {
                // Merge x^2 * x
                if *p1.base != *other {
                    return false;
                } else if let Term::Value(Value { value, ring, .. }) = &mut **p1.exposant {
                    // Add one to m1 coeff
                    ring.add_assign(value, &ring.one());
                    return true;
                } else {
                    **p1.exposant = Term::Add(Add::new(
                        vec![
                            (**p1.exposant).clone(),
                            Value::new(
                                other.get_set().get_exposant_set().one(),
                                other.get_set().get_exposant_set(),
                            )
                            .into(),
                        ],
                        other.get_set().get_exposant_set(),
                    ))
                    .into();
                    return true;
                }
            }
            (_, Term::Pow(p2)) => {
                // Merge x * x^2
                if *self != *p2.base {
                    return false;
                } else if let Term::Value(Value { value, ring, .. }) = &**p2.exposant {
                    // Transform self to mul and add one to coeff
                    self.to_pow(
                        p2.base.clone(),
                        Term::Value(Value::new(ring.add(&value, &ring.one()), *ring)).into(),
                    );
                    return true;
                } else {
                    // Add one to the coefficient
                    self.to_pow(
                        p2.base.clone(),
                        Term::Add(Add::new(
                            vec![
                                (**p2.exposant).clone(),
                                Value::new(
                                    self.get_set().get_exposant_set().one(),
                                    self.get_set().get_exposant_set(),
                                )
                                .into(),
                            ],
                            self.get_set().get_exposant_set(),
                        ))
                        .into(),
                    );
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
                    self.get_set().get_exposant_set().nth(2),
                    self.get_set().get_exposant_set(),
                )),
                self.get_set(),
            ));
            return true;
        }
        false
    }
    /// Convert `self` to [Mul], and return a mutable reference to easily edit it.
    fn to_mul(&mut self) -> &mut Mul<T> {
        *self = Term::Mul(Mul::new(vec![], self.get_set()));
        if let Term::Mul(n) = self {
            n
        } else {
            unreachable!()
        }
    }
    /// Convert `self` to [Pow], and return a mutable reference to easily edit it.
    fn to_pow(
        &mut self,
        base: Box<Term<T>>,
        exposant: Box<Term<<T as GroupImpl>::ExposantSet>>,
    ) -> &mut Pow<T> {
        *self = Term::Pow(Pow::new(base, exposant, self.get_set()));
        if let Term::Pow(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    /// Convert the expression to a multivariate polynomial.
    ///
    /// Used to use factorization algorithms.
    pub fn to_polynomial(&self) -> MultivariatePolynomial<Term<T>, T, TermField<T::ExposantSet>> {
        self.expand().to_polynomial_impl()
    }

    fn to_polynomial_impl(&self) -> MultivariatePolynomial<Term<T>, T, TermField<T::ExposantSet>> {
        match self {
            Term::Value(value) => MultivariatePolynomial::constant(
                value.clone().get_value(),
                self.get_set(),
                self.get_set().get_exposant_set().get_term_field(),
            ),
            Term::Symbol(_) | Term::Fun(_) => MultivariatePolynomial::variable(
                self.clone(),
                self.get_set(),
                self.get_set().get_exposant_set().get_term_field(),
            ),
            Term::Add(add) => {
                let mut res = MultivariatePolynomial::zero(
                    self.get_set(),
                    self.get_set().get_exposant_set().get_term_field(),
                );

                for term in add {
                    res = res + term.to_polynomial_impl()
                }

                res
            }
            Term::Mul(mul) => {
                let mut res = MultivariatePolynomial::one(
                    self.get_set(),
                    self.get_set().get_exposant_set().get_term_field(),
                );

                for factor in mul {
                    res = res * factor.to_polynomial_impl()
                }

                res
            }
            Term::Pow(pow) => MultivariatePolynomial::new(
                vec![(
                    vec![(*pow.base.clone(), (**pow.exposant).clone())],
                    self.get_set().one(),
                )],
                self.get_set(),
                self.get_set().get_exposant_set().get_term_field(),
            ),
        }
    }

    /// Check if the term is one
    /// ```
    /// use formalith::{field::R, parse, symbol};
    ///
    /// assert!(parse!("1", R).is_one());
    /// ```
    pub fn is_one(&self) -> bool {
        match self {
            Term::Value(value) => value.ring.is_one(&value.value),
            Term::Symbol(_) | Term::Add(_) | Term::Mul(_) | Term::Pow(_) => false,
            Term::Fun(_) => todo!(),
        }
    }

    /// Expand the expression
    /// ```
    /// use formalith::{field::R, parse, symbol};
    ///
    /// assert_eq!(parse!("(x+2)*(x+3)", R).expand(), parse!("x*x+5*x+6", R));
    /// ```
    pub fn expand(&self) -> Self {
        let res = self.expand_without_norm();
        let res = res.get_set().expand(res);
        res.normalize()
    }

    pub fn factor(&self) -> Self
    where
        Term<T>: TryFrom<Term<T::ExposantSet>>,
        T::Element: TryFrom<Term<T::ExposantSet>>,
        <Term<T> as TryFrom<Term<T::ExposantSet>>>::Error: Debug,
    {
        let poly = self.to_polynomial();
        let factors = poly.factor();
        let mut res = Term::one(self.get_set());
        for (factor, multiplicity) in factors {
            res *= factor.to_term().pow(&Term::constant(
                self.get_set().get_exposant_set().nth(multiplicity.into()),
                self.get_set().get_exposant_set(),
            ))
        }
        res
    }

    /// Expand the expression without normalizing it
    fn expand_without_norm(&self) -> Self {
        match self {
            Term::Value(_) => self.clone(),
            Term::Symbol(_) => self.clone(),
            Term::Add(add) => Term::Add(Add::new(
                add.iter().map(|term| term.expand()).collect(),
                add.ring,
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
                                    new_sums.push(Term::Mul(Mul::new(
                                        vec![sum.clone(), term.clone()],
                                        sum.get_set(),
                                    )))
                                }
                            }
                            if sums.is_empty() {
                                new_sums.push(term.clone());
                            }
                        }
                        std::mem::swap(&mut sums, &mut new_sums);
                        new_sums.clear();
                    } else if sums.is_empty() {
                        sums.push(expanded);
                    } else {
                        for sum in &mut sums {
                            if let Term::Mul(mul) = sum {
                                mul.push(expanded.clone());
                            } else {
                                *sum = Term::Mul(Mul::new(
                                    vec![sum.clone(), expanded.clone()],
                                    sum.get_set(),
                                ))
                            }
                        }
                    }
                }
                let res = Term::Add(Add::new(sums, mul.ring));
                res
            }
            Term::Pow(pow) => {
                let mut res = pow.clone();
                *res.base = (*res.base).expand();
                **res.exposant = (**res.exposant).expand();

                Term::Pow(res)
            }
            Term::Fun(fun) => fun.expand(),
        }
    }
    /// Normalize an expression, i.e. sort sums and products, merge expressions that can be merged, remove useless expressions like zero in sums.
    pub fn normalize(&self) -> Self {
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
                if res.len() == 0 {
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
                let res = if terms.len() == 1 {
                    terms.pop().unwrap()
                } else {
                    Term::Add(Add::new(terms, self.get_set()))
                };
                res
            }
            Term::Mul(mul) => {
                let mut res = vec![];
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
                    } else {
                        res.push(normalized);
                    }
                }
                if res.len() == 0 {
                    return Term::one(self.get_set());
                }
                res.sort_by(|a, b| a.cmp_factors(b));
                res.reverse();

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
                let res = if factors.len() == 1 {
                    factors.pop().unwrap()
                } else {
                    Term::Mul(Mul::new(factors, self.get_set()))
                };
                res
            }
            Term::Pow(pow) => {
                let base = (*pow.base).normalize();
                let exposant = (**pow.exposant).normalize();
                if exposant.is_one() {
                    return base;
                } else if exposant.is_zero() {
                    return Term::one(self.get_set());
                }
                Term::Pow(Pow::new(Box::new(base), Box::new(exposant), self.get_set()))
            }
            Term::Fun(fun) => fun.normalize(),
        };
        res = res.get_set().normalize(res);
        res.set_normalized(true);
        res
    }
    /// Return the expression as a rational expression, (numerator, denominator)
    pub fn as_fraction(&self, unify: bool) -> (Self, Self) {
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
                    **pow.exposant = pow.exposant.unify();
                }
                if pow.exposant.is_strictly_negative() {
                    **pow.exposant = -&**pow.exposant;
                    (Term::one(self.get_set()), pow.into())
                } else {
                    (pow.into(), Term::one(self.get_set()))
                }
            }
        }
    }
    /// Return the expression as a rational expression
    pub fn unify(&self) -> Self {
        let (num, den) = self.as_fraction(true);
        num / den
    }
    /// Simplify the expression by applying other simplification functions ([unify], [expand])
    pub fn simplify(&self) -> Self {
        let mut res = self.get_set().simplify(self.clone());
        let (num, den) = res.as_fraction(true);
        res = num.expand_without_norm() / den.expand_without_norm(); // add set expansion if expand_without_norm and add a value_expand arg ?
        res = res.normalize();

        res
    }
    /// Invert the expression
    pub fn inv(&self) -> Self {
        Term::one(self.get_set()) / self
    }
    /// Return the constant 1
    pub fn one(set: T) -> Self {
        Term::Value(Value::new(set.one(), set))
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

impl<T: Ring> Display for Term<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
impl<T: Ring> Print for Term<T> {
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
    fn pretty_print(&self, options: &PrintOptions) -> PrettyPrinter {
        match self {
            Term::Value(value) => Print::pretty_print(value, options),
            Term::Symbol(symbol) => Print::pretty_print(symbol, options),
            Term::Add(add) => Print::pretty_print(add, options),
            Term::Mul(mul) => Print::pretty_print(mul, options),
            Term::Pow(pow) => Print::pretty_print(pow, options),
            Term::Fun(fun) => Print::pretty_print(&**fun, options),
        }
    }
}
