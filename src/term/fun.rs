//! Function expression.

use std::{fmt::Debug, hash::Hash};

use dyn_clone::DynClone;
use dyn_eq::DynEq;
use dyn_hash::DynHash;
use owo_colors::colors::Cyan;

use crate::{
    context::{Context, Symbol},
    field::{Group, Ring},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term};

pub trait Function<T: Group>: Debug + DynClone + DynEq + DynHash + Flags + Print {
    fn normalize(&self) -> Term<T>;
    fn expand(&self) -> Term<T>;
    fn get_set(&self) -> T;
}

impl<From: Ring, T: Ring> Function<T> for Fun<From, T> {
    fn normalize(&self) -> Term<T> {
        let mut new_args = vec![];
        for arg in &self.args {
            new_args.push(arg.normalize());
        }
        Term::Fun(Box::new(Fun::new(self.ident, new_args, self.set)))
    }
    fn expand(&self) -> Term<T> {
        let mut new_args = vec![];
        for arg in &self.args {
            new_args.push(arg.expand());
        }
        Term::Fun(Box::new(Fun::new(self.ident, new_args, self.set)))
    }
    fn get_set(&self) -> T {
        self.set
    }
}

dyn_hash::hash_trait_object!(<T> Function<T>);
dyn_clone::clone_trait_object!(<T> Function<T> );
dyn_eq::eq_trait_object!(<T: 'static> Function<T>);

/// A function called in a mathematical expression, like `sqrt(x)`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fun<From: Group, T: Group = From> {
    flags: u8,
    pub(crate) ident: Symbol,
    pub(crate) args: Vec<Term<From>>,
    pub(crate) set: T,
}

impl<From: Ring, T: Ring> Fun<From, T> {
    pub fn get_ident(&self) -> &str {
        Context::get_symbol_data(&self.ident).name.as_str()
    }
}

impl<From: Group, T: Group> Fun<From, T> {
    /// Create a new function expression
    pub fn new(ident: Symbol, args: Vec<Term<From>>, set: T) -> Self {
        Self {
            flags: 0,
            ident,
            args,
            set,
        }
    }
}

impl<From: Ring, T: Ring> Flags for Fun<From, T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<From: Ring, T: Ring> Print for Fun<From, T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::fg::<Cyan>(self.get_ident(), options, f)?;
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
        let mut args = self.args.iter().map(|x| x.pretty_print(options));
        let mut res = args.next().unwrap();
        for mut arg in args {
            arg.left(' ');
            res.concat(",", false, &arg);
        }
        res.paren();
        match self.ident {
            Context::ABS => {
                let mut res = self.args.first().unwrap().pretty_print(options);
                res.group('|', '|');
                res
            }
            _ => {
                let mut fun = PrettyPrinter::from(format!("{}", self.get_ident(),));
                fun.concat("", false, &res);
                fun
            }
        }
    }
}
