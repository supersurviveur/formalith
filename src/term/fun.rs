//! Function expression.

use std::{fmt::Debug, hash::Hash};

use dyn_clone::DynClone;
use dyn_eq::DynEq;
use dyn_hash::DynHash;
use owo_colors::colors::Cyan;

use crate::{
    context::{Context, Symbol},
    field::{GroupBound, RingBound},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term};

/// To allow function from any set to the set `T`, a dynamic dispatch is used with this trait, implementing needed methods.
///
/// See [Fun] for a structure implementing this trait.
pub trait Function<T: GroupBound>: Debug + DynClone + DynEq + DynHash + Flags + Print {
    /// Normalize arguments of the function.
    fn normalize(&self) -> Term<T>;
    /// Expand arguments of the function.
    fn expand(&self) -> Term<T>;
    /// Get the set where the function lives.
    fn get_set(&self) -> T;
    /// Get the identifier of the function.
    fn get_ident(&self) -> Symbol;
    /// Get the identifier of the function as a string.
    fn get_ident_as_str(&self) -> &str {
        Context::get_symbol_data(&self.get_ident()).name.as_str()
    }
}

impl<T: GroupBound> dyn Function<T> {
    /// Try to get arguments of the function in set `From`
    pub fn get_args<From: GroupBound>(&self) -> Option<&Vec<Term<From>>> {
        self.as_any()
            .downcast_ref::<Fun<From, T>>()
            .map(|fun| &fun.args)
    }
    /// Try to get first argument of the function in set `From`
    pub fn get_arg<From: GroupBound>(&self) -> Option<&Term<From>> {
        self.as_any()
            .downcast_ref::<Fun<From, T>>()
            .map(|fun| fun.args.first())?
    }
}

impl<From: RingBound, T: RingBound> Function<T> for Fun<From, T> {
    fn normalize(&self) -> Term<T> {
        let mut new_args = vec![];
        for arg in &self.args {
            new_args.push(arg.normalize());
        }
        Term::Fun(Box::new(Fun::new(self.ident, new_args, self.set)) as Box<dyn Function<T>>)
    }
    fn expand(&self) -> Term<T> {
        let mut new_args = vec![];
        for arg in &self.args {
            new_args.push(arg.expand());
        }
        Term::Fun(Box::new(Fun::new(self.ident, new_args, self.set)) as Box<dyn Function<T>>)
    }
    fn get_set(&self) -> T {
        self.set
    }
    fn get_ident(&self) -> Symbol {
        self.ident
    }
}

dyn_hash::hash_trait_object!(<T> Function<T>);
dyn_clone::clone_trait_object!(<T> Function<T> );
dyn_eq::eq_trait_object!(<T: 'static> Function<T>);

/// A function called in a mathematical expression, like `sqrt(x)`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fun<From: GroupBound, T: GroupBound = From> {
    flags: u8,
    pub(crate) ident: Symbol,
    pub(crate) args: Vec<Term<From>>,
    pub(crate) set: T,
}

impl<From: GroupBound, T: GroupBound> Fun<From, T> {
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

impl<From: RingBound, T: RingBound> Flags for Fun<From, T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<From: RingBound, T: RingBound> Print for Fun<From, T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::fg::<Cyan>(self.get_ident_as_str(), options, f)?;
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
                let mut fun = PrettyPrinter::from(self.get_ident_as_str().to_string());
                fun.concat("", false, &res);
                fun
            }
        }
    }
}
