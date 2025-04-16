use std::{fmt::Display, mem::ManuallyDrop};

use crate::field::{Field, Group, Ring, Z};

use super::{Flags, Term, Value};

/// A product of expressions.
#[derive(Clone, Debug, PartialEq)]
pub struct Pow<T: Group> {
    flags: u8,
    pub(crate) base: Box<Term<T>>,
    /// `ManuallyDrop` is used to avoid drop-checking which fails for infinite type `T::ExposantSet::ExposantSet::...`
    pub(crate) exposant: ManuallyDrop<Box<Term<T::ExposantSet>>>,
    pub(crate) ring: &'static T,
}

impl<T: Group> Flags for Pow<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Group> Pow<T> {
    pub fn new<E: Into<Box<Term<T>>>, F: Into<Box<Term<T::ExposantSet>>>>(
        base: E,
        exposant: F,
        ring: &'static T,
    ) -> Self {
        Self {
            flags: 0,
            base: base.into(),
            exposant: ManuallyDrop::new(exposant.into()),
            ring,
        }
    }
}

impl<T: Group> Drop for Pow<T> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.exposant);
        }
    }
}

impl<T: Group> Display for Pow<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let need_paren = match *self.base {
            Term::Value(_) | Term::Symbol(_) | Term::Fun(_) => false,
            Term::Mul(_) | Term::Add(_) | Term::Pow(_) => true,
        };
        if need_paren {
            write!(f, "({})", self.base)?;
        } else {
            write!(f, "{}", self.base)?;
        }
        write!(f, "^")?;
        let need_paren = match **self.exposant {
            Term::Value(_) | Term::Symbol(_) | Term::Fun(_) => false,
            Term::Mul(_) | Term::Add(_) | Term::Pow(_) => true,
        };
        if need_paren {
            write!(f, "({})", *self.exposant)?;
        } else {
            write!(f, "{}", *self.exposant)?;
        }
        Ok(())
    }
}
