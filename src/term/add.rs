//! Sum expression.

use std::fmt::Display;

use crate::{
    field::{GroupBound, Ring, RingBound},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term, Value};

/// A sum of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Add<T: GroupBound> {
    flags: u8,
    pub(crate) terms: Vec<Term<T>>,
    pub(crate) ring: T,
}

impl<T: Ring> Flags for Add<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Ring> Add<T> {
    /// Create an empty sum expression.
    pub fn new(terms: Vec<Term<T>>, ring: T) -> Self {
        Self {
            flags: 0,
            terms,
            ring,
        }
    }
    /// Create a new empty sum, with an inner vec with a defined capacity.
    pub fn with_capacity(capacity: usize, ring: T) -> Self {
        Self {
            flags: 0,
            terms: Vec::with_capacity(capacity),
            ring,
        }
    }
    /// Get the number of terms in the sum.
    pub fn len(&self) -> usize {
        self.terms.len()
    }
    /// Add a term at the end of the sum, without normalizing it
    pub fn push(&mut self, value: Term<T>) {
        self.set_normalized(false);
        self.terms.push(value);
    }
    /// Return an iterator over the sum terms
    pub fn iter(&self) -> std::slice::Iter<'_, Term<T>> {
        self.terms.iter()
    }
}

impl<T: RingBound> IntoIterator for Add<T> {
    type Item = Term<T>;

    type IntoIter = std::vec::IntoIter<Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}

impl<'a, T: RingBound> IntoIterator for &'a Add<T> {
    type Item = &'a Term<T>;

    type IntoIter = std::slice::Iter<'a, Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<T: RingBound> Print for Add<T> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        for (i, term) in self.terms.iter().enumerate() {
            match term {
                Term::Value(Value { value, ring, .. }) if term.is_strictly_negative() => {
                    if i != 0 {
                        Self::operator("-", options, f)?;
                    }
                    Print::print(&Value::new(ring.neg(value), *ring), options, f)?;
                }
                Term::Mul(mul)
                    if mul.has_coeff()
                        && mul.factors.last().unwrap().is_strictly_negative()
                        && i != 0 =>
                {
                    let mut mul = mul.clone();
                    let last = mul.factors.last_mut().unwrap();
                    mul.ring.neg_assign(&mut unsafe { last.as_value() }.value);
                    Self::operator("-", options, f)?;
                    Print::print(&mul, options, f)?;
                }
                _ => {
                    if i != 0 {
                        Self::operator("+", options, f)?;
                    }
                    Print::print(term, options, f)?;
                }
            }
        }
        Ok(())
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        let mut res: Option<PrettyPrinter> = None;
        for term in self.terms.iter() {
            match term {
                Term::Value(Value { value, ring, .. }) if term.is_strictly_negative() => {
                    let elem = Print::pretty_print(&Value::new(ring.neg(value), *ring), options);
                    if let Some(res) = &mut res {
                        res.concat("-", true, &elem);
                    } else {
                        res = Some(elem);
                    }
                }
                Term::Mul(mul)
                    if {
                        match mul.factors.last() {
                            Some(Term::Value(_)) => {
                                mul.factors.last().unwrap().is_strictly_negative()
                            }
                            _ => false,
                        }
                    } && res.is_some() =>
                {
                    let mut mul = mul.clone();
                    let last = mul.factors.last_mut().unwrap();
                    mul.ring.neg_assign(&mut unsafe { last.as_value() }.value);
                    let elem = Print::pretty_print(&mul, options);
                    res.as_mut().unwrap().concat("-", true, &elem);
                }
                _ => {
                    let elem = Print::pretty_print(term, options);
                    if let Some(res) = &mut res {
                        res.concat("+", true, &elem);
                    } else {
                        res = Some(elem);
                    }
                }
            }
        }
        res.unwrap()
    }
}

impl<T: RingBound> Display for Add<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
