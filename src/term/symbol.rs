//! Symbol expression.

use std::fmt::{Debug, Display};

use crate::{
    context::{Context, Symbol},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, NORMALIZED};

/// A symbol inside a mathematical expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolTerm<T> {
    flags: u8,
    pub(crate) symbol: Symbol,
    pub(crate) ring: T,
}

impl<T> Flags for SymbolTerm<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T> SymbolTerm<T> {
    /// Create a new symbol expression
    pub fn new(symbol: Symbol, ring: T) -> Self {
        Self {
            flags: NORMALIZED,
            symbol,
            ring,
        }
    }
}
impl<T: Debug> Print for SymbolTerm<T> {
    fn print(&self, _options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Context::get_symbol_data(&self.symbol).name)
    }

    fn pretty_print(&self, _options: &PrintOptions) -> crate::printer::PrettyPrinter {
        PrettyPrinter::from(Context::get_symbol_data(&self.symbol).name.clone())
    }
}
impl<T: Debug> Display for SymbolTerm<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
