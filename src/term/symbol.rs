use std::fmt::Display;

use crate::{
    context::{Context, Symbol},
    field::{Field, Group, Ring},
};

use super::{Flags, NORMALIZED};

/// A symbol inside a mathematical expression.
#[derive(Clone, Debug, PartialEq)]
pub struct SymbolTerm<T: Group> {
    flags: u8,
    pub(crate) symbol: Symbol,
    pub(crate) ring: &'static T,
}

impl<T: Group> Flags for SymbolTerm<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Group> SymbolTerm<T> {
    pub fn new(symbol: Symbol, ring: &'static T) -> Self {
        Self {
            flags: NORMALIZED,
            symbol,
            ring,
        }
    }
}
impl<T: Group> Display for SymbolTerm<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Context::get_symbol_data(&self.symbol).name)
    }
}
