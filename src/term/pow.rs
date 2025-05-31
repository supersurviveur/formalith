//! Power expression.

use std::{fmt::Display, mem::ManuallyDrop};

use crate::{
    field::{GroupBound, RingBound},
    printer::{Print, PrintOptions},
};

use super::{Flags, Term};

/// A product of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pow<T: GroupBound> {
    flags: u8,
    pub(crate) base: Box<Term<T>>,
    /// `ManuallyDrop` is used to avoid drop-checking which fails for infinite type `T::ExposantSet::ExposantSet::...`
    pub(crate) exposant: ManuallyDrop<Box<Term<T::ExposantSet>>>,
    pub(crate) set: T,
}

impl<T: RingBound> Flags for Pow<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: RingBound> Pow<T> {
    /// Create a new pow expression
    pub fn new<E: Into<Box<Term<T>>>, F: Into<Box<Term<T::ExposantSet>>>>(
        base: E,
        exposant: F,
        ring: T,
    ) -> Self {
        Self {
            flags: 0,
            base: base.into(),
            exposant: ManuallyDrop::new(exposant.into()),
            set: ring,
        }
    }
}

impl<T: GroupBound> Drop for Pow<T> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.exposant);
        }
    }
}

impl<T: RingBound> Print for Pow<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::group(&self.base, options, f)?;
        Self::operator("^", options, f)?;
        Self::group(&**self.exposant, options, f)?;
        Ok(())
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        // If exposant is negative, print as a fraction
        let pow = match **self.exposant {
            Term::Value(_) if self.exposant.is_strictly_negative() => {
                let mut pow = self.clone();
                **pow.exposant = -&**pow.exposant;
                match Term::Pow(pow.clone()).normalize() {
                    Term::Pow(pow) => pow,
                    normalized => {
                        let mut res = Print::pretty_print(&Term::one(self.set), options);
                        res.vertical_concat("─", &Print::pretty_print(&normalized, options));
                        return res;
                    }
                }
            }
            _ => self.clone(),
        };
        let mut base = (*self.base).pretty_print(options);
        let exposant = (**pow.exposant).pretty_print(options);
        if match *self.base {
            Term::Mul(_) | Term::Add(_) => true,
            _ => false,
        } {
            base.paren();
        }
        base.pow(&exposant);
        match **self.exposant {
            Term::Value(_) if self.exposant.is_strictly_negative() => {
                let mut res = Print::pretty_print(&Term::one(self.set), options);
                res.vertical_concat("─", &base);
                res
            }
            _ => base,
        }
    }
}
impl<T: RingBound> Display for Pow<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
