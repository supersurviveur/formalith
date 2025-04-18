//! Sum expression.

use std::fmt::Display;

use crate::{
    field::Group,
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term};

/// A sum of expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Add<T: Group> {
    flags: u8,
    pub(crate) terms: Vec<Term<T>>,
    pub(crate) ring: &'static T,
}

impl<T: Group> Flags for Add<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Group> Add<T> {
    /// Create an empty sum expression.
    pub fn new(terms: Vec<Term<T>>, ring: &'static T) -> Self {
        Self {
            flags: 0,
            terms,
            ring,
        }
    }
    /// Create a new empty sum, with an inner vec with a defined capacity.
    pub fn with_capacity(capacity: usize, ring: &'static T) -> Self {
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

impl<T: Group> Print for Add<T> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        for (i, term) in self.terms.iter().enumerate() {
            Print::print(term, options, f)?;
            if i != self.len() - 1 {
                Self::operator("+", options, f)?;
            }
        }
        Ok(())
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        let mut res: Option<PrettyPrinter> = None;
        for term in self.terms.iter() {
            let elem = Print::pretty_print(term, options);
            if let Some(res) = &mut res {
                res.concat("+", &elem);
            } else {
                res = Some(elem);
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
