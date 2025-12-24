//! Sum expression.

use std::fmt::Display;

use crate::{
    field::{Group, Set},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term, Value};

/// A sum of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Add<T: Set> {
    flags: u8,
    pub(crate) terms: Vec<Term<T>>,
    pub(crate) set: T,
}

impl<T: Set> Flags for Add<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Set> Add<T> {
    /// Create an empty sum expression.
    pub fn new(terms: Vec<Term<T>>, set: T) -> Self {
        Self {
            flags: 0,
            terms,
            set,
        }
    }
    /// Create a new empty sum, with an inner vec with a defined capacity.
    pub fn with_capacity(capacity: usize, set: T) -> Self {
        Self {
            flags: 0,
            terms: Vec::with_capacity(capacity),
            set,
        }
    }
    /// Returns true if the addition is empty.
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
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

impl<T: Set> IntoIterator for Add<T> {
    type Item = Term<T>;

    type IntoIter = std::vec::IntoIter<Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}

impl<'a, T: Set> IntoIterator for &'a Add<T> {
    type Item = &'a Term<T>;

    type IntoIter = std::slice::Iter<'a, Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<T: Group> Print for Add<T> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        for (i, term) in self.terms.iter().enumerate() {
            match term {
                Term::Value(Value { value, set, .. }) if term.is_strictly_negative() => {
                    if i != 0 {
                        Self::operator("-", options, f)?;
                    }
                    Print::print(&Value::new(set.neg(value), *set), options, f)?;
                }
                Term::Mul(mul)
                    if mul
                        .set
                        .get_coefficient_set()
                        .is_strictly_negative(&mul.get_coeff())
                        && i != 0 =>
                {
                    let mut mul = mul.clone();
                    let coeff = &mut mul.coefficient;
                    mul.set.get_coefficient_set().neg_assign(coeff);
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
                Term::Value(Value { value, set, .. }) if term.is_strictly_negative() => {
                    let elem = Print::pretty_print(&Value::new(set.neg(value), *set), options);
                    if let Some(res) = &mut res {
                        res.concat("-", true, &elem);
                    } else {
                        res = Some(elem);
                    }
                }
                Term::Mul(mul)
                    if mul
                        .set
                        .get_coefficient_set()
                        .is_strictly_negative(&mul.get_coeff())
                        && res.is_some() =>
                {
                    let mut mul = mul.clone();
                    let coeff = &mut mul.coefficient;
                    mul.set.get_coefficient_set().neg_assign(coeff);
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

impl<T: Group> Display for Add<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
