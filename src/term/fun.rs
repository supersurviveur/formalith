//! Function expression.

use std::fmt::Display;

use owo_colors::colors::Cyan;

use crate::{
    context::{Context, Symbol},
    field::Group,
    printer::{PrettyPrinter, Print, PrintOptions},
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
        match self.ident {
            Context::ABS => {
                let mut res = self.args.first().unwrap().pretty_print(options);
                res.group('|', '|');
                res
            }
            _ => {
                let mut args = self.args.iter().map(|x| x.pretty_print(options));
                let mut res = args.next().unwrap();
                for mut arg in args {
                    arg.left(' ');
                    res.concat(",", false, &arg);
                }
                res.paren();
                let mut fun = PrettyPrinter::from(format!(
                    "{}",
                    Context::get_symbol_data(&self.ident).name.as_str(),
                ));
                fun.concat("", false, &res);
                fun
            }
        }
    }
}

impl<T: Group> Display for Fun<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
