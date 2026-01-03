//! Constant value expression.

use crate::{
    field::Set,
    printer::{PrettyPrint, Print},
};

use super::{Flags, NORMALIZED};

/// A constant in a mathematical expression, living in the algebraic set T
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::struct_field_names)]
pub struct Value<T: Set, V = <T as Set>::Element> {
    flags: u8,
    pub(crate) value: V,
    pub(crate) set: T,
}

impl<T: Set> Value<T> {
    /// Create a new `Value` from an element of the set T
    pub fn new(value: T::Element, set: T) -> Self {
        Self {
            flags: NORMALIZED,
            value,
            set,
        }
    }
    /// Get the inner constant
    pub fn get_value(self) -> T::Element {
        self.value
    }
}

impl<T: Set> Flags for Value<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Set> Print for Value<T> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        self.set.print(&self.value, options, f)
    }
}
impl<T: Set> PrettyPrint for Value<T> {
    fn pretty_print(
        &self,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        self.set.pretty_print(&self.value, options)
    }
}
