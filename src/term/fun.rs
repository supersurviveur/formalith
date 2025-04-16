use std::fmt::Display;

use crate::{
    context::{Context, Symbol},
    field::{Field, Group, Ring},
};

use super::{Flags, Term, NORMALIZED};

/// A function called in a mathematical expression, like `sqrt(x)`
#[derive(Clone, Debug, PartialEq)]
pub struct Fun<T: Group> {
    flags: u8,
    pub(crate) ident: Symbol,
    pub(crate) args: Vec<Term<T>>,
    pub(crate) ring: &'static T,
}

impl<T: Group> Fun<T> {
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

impl<T: Group> Display for Fun<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", Context::get_symbol_data(&self.ident).name)?;
        for (i, term) in self.args.iter().enumerate() {
            if i != self.args.len() - 1 {
                write!(f, "{}, ", term)?;
            } else {
                write!(f, "{}", term)?;
            }
        }
        write!(f, ")")
    }
}
