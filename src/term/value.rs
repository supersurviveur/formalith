use crate::field::{Field, Group, Ring};

use super::{Flags, NORMALIZED};

/// A constant in a mathematical expression, living in the algebraic set T
#[derive(Clone, Debug, PartialEq)]
pub struct Value<T: Group> {
    flags: u8,
    pub(crate) value: T::Element,
    pub(crate) ring: &'static T,
}

impl<T: Group> Value<T> {
    pub fn new(value: T::Element, ring: &'static T) -> Self {
        Self {
            flags: NORMALIZED,
            value,
            ring,
        }
    }
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
