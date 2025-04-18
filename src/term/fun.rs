//! Function expression.

use std::fmt::Display;

use owo_colors::colors::{Blue, Cyan};

use crate::{
    context::{Context, Symbol},
    field::Group,
    printer::{Print, PrintOptions},
};

use super::{Flags, Term};

/// A function called in a mathematical expression, like `sqrt(x)`
#[derive(Clone, Debug, PartialEq)]
pub struct Fun<T: Group> {
    flags: u8,
    pub(crate) ident: Symbol,
    pub(crate) args: Vec<Term<T>>,
    pub(crate) ring: &'static T,
}

impl<T: Group> Fun<T> {
    /// Create a new function expression
    pub fn new(ident: Symbol, args: Vec<Term<T>>, ring: &'static T) -> Self {
        Self {
            flags: 0,
            ident,
            args,
            ring,
        }
    }
}

impl<T: Group> Flags for Fun<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Group> Print for Fun<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::fg::<Cyan>(
            Context::get_symbol_data(&self.ident).name.as_str(),
            options,
            f,
        )?;
        Self::group_delim("(", options, f)?;
        for (i, term) in self.args.iter().enumerate() {
            write!(f, "{}", term)?;
            if i != self.args.len() - 1 {
                Self::delimiter(", ", options, f)?;
            }
        }
        Self::group_delim(")", options, f)
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        todo!()
    }
}

impl<T: Group> Display for Fun<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
