//! Product expression.

use std::fmt::Display;

use crate::{
    field::{Group, Ring},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term, Value};

/// A product of expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Mul<T: Group> {
    flags: u8,
    pub(crate) factors: Vec<Term<T>>,
    pub(crate) ring: &'static T,
}

impl<T: Group> Flags for Mul<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Group> Mul<T> {
    /// Create a new product expression
    pub fn new(factors: Vec<Term<T>>, ring: &'static T) -> Self {
        Self {
            flags: 0,
            factors,
            ring,
        }
    }
    /// Create a new empty product, with an inner Vec with a defined capacity.
    pub fn with_capacity(capacity: usize, ring: &'static T) -> Self {
        Self {
            flags: 0,
            factors: Vec::with_capacity(capacity),
            ring,
        }
    }
    /// Get the number of terms in the sum.
    pub fn len(&self) -> usize {
        self.factors.len()
    }
    /// Add a factor at the end of the product, without normalizing it
    pub fn push(&mut self, value: Term<T>) {
        self.set_normalized(false);
        self.factors.push(value);
    }
    /// Extend the product with another one, without normalizing it
    pub fn extend(&mut self, other: Mul<T>) {
        self.set_normalized(false);
        self.factors.extend(other.factors);
    }
    /// Return an iterator over the product factors.
    pub fn iter(&self) -> std::slice::Iter<'_, Term<T>> {
        self.factors.iter()
    }
}

impl<T: Ring> Mul<T> {
    /// Return the constant coefficient factor of the product if it exists, one otherwise.
    pub fn get_coeff(&self) -> T::Element {
        if let Term::Value(v) = &self.factors.last().unwrap() {
            v.value.clone()
        } else {
            self.ring.one()
        }
    }
    /// Checks if the product have a constant coefficient.
    pub fn has_coeff(&self) -> bool {
        if let Term::Value(_) = &self.factors.last().unwrap() {
            true
        } else {
            false
        }
    }
    /// Set the constant coefficient factor of the product
    pub fn set_coeff(&mut self, coeff: T::Element) {
        if let Some(Term::Value(v)) = self.factors.last_mut() {
            v.value = coeff
        } else {
            self.factors.push(Term::Value(Value::new(coeff, self.ring)));
        }
    }
}

impl<T: Group> Print for Mul<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, factor) in self.factors.iter().enumerate() {
            Self::group(factor, options, f)?;
            if i != self.len() - 1 {
                Self::operator("*", options, f)?;
            }
        }
        Ok(())
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        let mut res: Option<PrettyPrinter> = None;
        for term in self.factors.iter() {
            let elem = Print::pretty_print(term, options);
            if let Some(res) = &mut res {
                res.concat("*", &elem);
            } else {
                res = Some(elem);
            }
        }
        res.unwrap()
    }
}

impl<T: Group> Display for Mul<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
