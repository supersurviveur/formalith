//! Mathematical expression implementation. An expression is represented as a [Term], which can be of many types.
//! A term lives in a specific set, which can be a group, a ring, a field... Standard operations are defined between terms in [term_op].

use std::{borrow::BorrowMut, cmp::Ordering, fmt::Display};

use crate::{
    context::{Context, Symbol},
    field::{Group, Ring},
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
#[derive(Clone, Debug, PartialEq)]
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
    Fun(Fun<T>),
}

// #[derive(Clone, Debug, PartialEq)]
// pub struct MultivariatePolynomial<T: Ring, U: Ring = T> {
//     terms: Vec<(Vec<(Symbol, U::Element)>, T::Element)>,
//     ring: T,
// }

impl<T: Group> std::cmp::PartialOrd for Term<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Term::Value(Value { value: v1, .. }), Term::Value(Value { value: v2, .. })) => {
                v1.partial_cmp(v2)
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

impl<T: Group> Flags for Term<T> {
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

impl<T: Group> From<Mul<T>> for Term<T> {
    fn from(value: Mul<T>) -> Self {
        Term::Mul(value)
    }
}

impl<T: Group> From<Add<T>> for Term<T> {
    fn from(value: Add<T>) -> Self {
        Term::Add(value)
    }
}

impl<T: Group> From<Value<T>> for Term<T> {
    fn from(value: Value<T>) -> Self {
        Term::Value(value)
    }
}

impl<T: Group> From<Pow<T>> for Term<T> {
    fn from(value: Pow<T>) -> Self {
        Term::Pow(value)
    }
}

impl<T: Group> Term<T> {
    /// Absolute function
    pub const ABS: Symbol = Context::ABS;

    /// Create a new empty product expression
    pub fn new_mul(ring: &'static T) -> Self {
        Term::Mul(Mul::new(vec![], ring))
    }
    /// Get the ring where constants live
    pub fn get_set(&self) -> &'static T {
        match self {
            Term::Value(value) => value.ring,
            Term::Symbol(symbol) => symbol.ring,
            Term::Add(add) => add.ring,
            Term::Mul(mul) => mul.ring,
            Term::Pow(pow) => pow.ring,
            Term::Fun(fun) => fun.ring,
        }
    }
}

impl<T: Group> Term<T> {
    /// Compare two terms
    pub fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Term::Value(Value { value: v1, .. }), Term::Value(Value { value: v2, .. })) => {
                v1.partial_cmp(v2).unwrap_or(Ordering::Equal)
            }
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
                            p1.ring.add(&p1.ring.one(), &p1.ring.one()), // Use a integer to T conversion in Ring Trait instead ?
                            p1.ring,
                        )),
                    ],
                    p1.ring,
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
        exposant: Box<Term<<T as Group>::ExposantSet>>,
    ) -> &mut Pow<T> {
        *self = Term::Pow(Pow::new(base, exposant, self.get_set()));
        if let Term::Pow(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    /// Return the zero constant expression
    pub fn zero(set: &'static T) -> Self {
        Term::Value(Value::new(set.zero(), set))
    }

    /// Check if the term is zero
    /// ```
    /// use formalith::{field::C, parse, symbol};
    ///
    /// assert!(parse!("0", C).is_zero());
    /// ```
    pub fn is_zero(&self) -> bool {
        match self {
            Term::Value(value) => value.ring.is_zero(&value.value),
            Term::Symbol(_) | Term::Add(_) | Term::Mul(_) | Term::Pow(_) => false,
            Term::Fun(_) => todo!(),
        }
    }

    /// Check if the term is one
    /// ```
    /// use formalith::{field::C, parse, symbol};
    ///
    /// assert!(parse!("1", C).is_one());
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
    /// use formalith::{field::C, parse, symbol};
    ///
    /// assert_eq!(parse!("(x+2)*(x+3)", C).expand(), parse!("x*x+5*x+6", C));
    /// ```
    pub fn expand(&self) -> Self {
        let res = self.expand_without_norm();
        res.normalize()
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
                *res.base = res.base.expand();
                **res.exposant = (*res.exposant).expand();

                Term::Pow(res)
            }
            Term::Fun(fun) => {
                let mut new_args = vec![];
                for arg in &fun.args {
                    new_args.push(arg.expand());
                }
                Term::Fun(Fun::new(fun.ident, new_args, fun.ring))
            }
        }
    }
    /// Normalize an expression, i.e. sort sums and products, merge expressions that can be merged, remove useless expressions like zero in sums.
    pub fn normalize(&self) -> Self {
        // return self.clone();
        if !self.needs_normalization() {
            return self.clone();
        }
        let mut res = match self {
            Term::Value(_) => self.clone(),
            Term::Symbol(_) => self.clone(),
            Term::Add(add) => {
                let mut res = vec![];
                for term in add.iter() {
                    let normalized = term.normalize();
                    if let Term::Add(inner_add) = normalized {
                        for term in inner_add.terms {
                            // terms are already normalized
                            res.push(term);
                        }
                    } else {
                        res.push(normalized);
                    }
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
                    if let Term::Mul(inner_mul) = normalized {
                        for factor in inner_mul.factors {
                            // factors are already normalized
                            res.push(factor);
                        }
                    } else {
                        res.push(normalized);
                    }
                }
                res.sort_by(|a, b| a.cmp_factors(b));
                res.reverse();

                // Merge factors
                let mut factors = vec![];
                let mut current_merge = res.pop().unwrap();
                while let Some(next) = res.pop() {
                    if !current_merge.merge_factors(&next) {
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
                Term::Pow(Pow::new(Box::new(base), Box::new(exposant), self.get_set()))
            }
            Term::Fun(fun) => {
                let mut new_args = vec![];
                for arg in &fun.args {
                    new_args.push(arg.normalize());
                }
                Term::Fun(Fun::new(fun.ident, new_args, fun.ring))
            }
        };
        res = res.get_set().normalize(res);
        res.set_normalized(true);
        res
    }
    /// Invert the expression
    pub fn inv(&self) -> Self {
        self.one() / self
    }
    /// Return the constant 1
    pub fn one(&self) -> Self {
        Term::Value(Value::new(self.get_set().one(), self.get_set()))
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

impl<T: Group> Display for Term<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
impl<T: Group> Print for Term<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Value(value) => Print::print(value, options, f),
            Term::Symbol(symbol) => Print::print(symbol, options, f),
            Term::Add(add) => Print::print(add, options, f),
            Term::Mul(mul) => Print::print(mul, options, f),
            Term::Pow(pow) => Print::print(pow, options, f),
            Term::Fun(fun) => Print::print(fun, options, f),
        }
    }
    fn pretty_print(&self, options: &PrintOptions) -> PrettyPrinter {
        match self {
            Term::Value(value) => Print::pretty_print(value, options),
            Term::Symbol(symbol) => Print::pretty_print(symbol, options),
            Term::Add(add) => Print::pretty_print(add, options),
            Term::Mul(mul) => Print::pretty_print(mul, options),
            Term::Pow(pow) => Print::pretty_print(pow, options),
            Term::Fun(fun) => Print::pretty_print(fun, options),
        }
    }
}
