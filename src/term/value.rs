//! Constant value expression.

use crate::{
    field::{Group, Ring},
    printer::Print,
};

use super::{Flags, NORMALIZED};

/// A constant in a mathematical expression, living in the algebraic set T
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Value<T: Group> {
    flags: u8,
    pub(crate) value: T::Element,
    pub(crate) ring: T,
}

impl<T: Group> Value<T> {
    /// Create a new `Value` from an element of the set T
    pub fn new(value: T::Element, ring: T) -> Self {
        Self {
            flags: NORMALIZED,
            value,
            ring,
        }
    }
    /// Get the inner constant
    pub fn get_value(self) -> T::Element {
        self.value
    }
}

impl<T: Group> Flags for Value<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Ring> Print for Value<T> {
    fn print(
        &self,
        _options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }

    fn pretty_print(
        &self,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        self.ring.pretty_print(&self.value, options)
    }
}
