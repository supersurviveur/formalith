//! Power expression.

use std::{fmt::Display, mem::ManuallyDrop};

use crate::{
    field::{Ring, Set},
    printer::{PrettyPrint, Print, PrintOptions},
    term::Normalize,
};

use super::{Flags, Term};

/// A product of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Pow<T: Set, E: Set = <T as Set>::ExponantSet> {
    flags: u8,
    pub(crate) base: Box<Term<T>>,
    /// `ManuallyDrop` is used to avoid drop-checking which fails for infinite type `T::ExponantSet::ExponantSet::...`
    pub(crate) exponant: ManuallyDrop<Box<Term<E>>>,
    pub(crate) set: T,
}

impl<T: Set> Flags for Pow<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Set> Pow<T> {
    /// Create a new pow expression
    pub fn new<E: Into<Box<Term<T>>>, F: Into<Box<Term<T::ExponantSet>>>>(
        base: E,
        exponant: F,
        set: T,
    ) -> Self {
        Self {
            flags: 0,
            base: base.into(),
            exponant: ManuallyDrop::new(exponant.into()),
            set,
        }
    }
}

impl<T: Set, E: Set> Drop for Pow<T, E> {
    fn drop(&mut self) {
        unsafe {
            ManuallyDrop::drop(&mut self.exponant);
        }
    }
}
impl<T: Set> Print for Pow<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Self::group(&self.base, options, f)?;
        Self::operator("^", options, f)?;
        Self::group(&**self.exponant, options, f)?;
        Ok(())
    }
}

impl<T: Set> PrettyPrint for Pow<T> {
    default fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        let pow = match **self.exponant {
            Term::Value(_) if self.exponant.is_strictly_negative() => {
                let mut pow = self.clone();
                **pow.exponant = -&**pow.exponant;
                pow
            }
            _ => self.clone(),
        };
        let mut base = (*self.base).pretty_print(options);
        let exponant = (**pow.exponant).pretty_print(options);
        if matches!(*self.base, Term::Mul(_) | Term::Add(_)) {
            base.paren();
        }
        base.pow(&exponant);
        base
    }
}

impl<T: Ring> PrettyPrint for Pow<T> {
    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        // If exponant is negative, print as a fraction
        let pow = match **self.exponant {
            Term::Value(_) if self.exponant.is_strictly_negative() => {
                let mut pow = self.clone();
                **pow.exponant = -&**pow.exponant;
                match Term::Pow(pow.clone()).normalize() {
                    Term::Pow(pow) => pow,
                    normalized => {
                        let mut res = PrettyPrint::pretty_print(&Term::one(self.set), options);
                        res.vertical_concat("─", &PrettyPrint::pretty_print(&normalized, options));
                        return res;
                    }
                }
            }
            _ => self.clone(),
        };
        let mut base = (*self.base).pretty_print(options);
        let exponant = (**pow.exponant).pretty_print(options);
        if matches!(*self.base, Term::Mul(_) | Term::Add(_)) {
            base.paren();
        }
        base.pow(&exponant);
        match **self.exponant {
            Term::Value(_) if self.exponant.is_strictly_negative() => {
                let mut res = PrettyPrint::pretty_print(&Term::one(self.set), options);
                res.vertical_concat("─", &base);
                res
            }
            _ => base,
        }
    }
}

impl<T: Set> Display for Pow<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        PrettyPrint::fmt(self, &PrintOptions::default(), f)
    }
}
