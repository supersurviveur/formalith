//! Product expression.

use std::fmt::Display;

use crate::{
    field::{GroupBound, RingBound},
    printer::{PrettyPrinter, Print, PrintOptions},
};

use super::{Flags, Term, Value};

/// A product of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mul<T: GroupBound> {
    flags: u8,
    pub(crate) factors: Vec<Term<T>>,
    pub(crate) ring: T,
}

impl<T: RingBound> Flags for Mul<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: RingBound> Mul<T> {
    /// Create a new product expression
    pub fn new(factors: Vec<Term<T>>, ring: T) -> Self {
        Self {
            flags: 0,
            factors,
            ring,
        }
    }
    /// Create a new empty product, with an inner Vec with a defined capacity.
    pub fn with_capacity(capacity: usize, ring: T) -> Self {
        Self {
            flags: 0,
            factors: Vec::with_capacity(capacity),
            ring,
        }
    }
    /// Get the number of terms in the sum.
    pub fn len(&self) -> usize {
        self.factors.len()
    }
    /// Add a factor at the end of the product, without normalizing it
    pub fn push(&mut self, value: Term<T>) {
        self.set_normalized(false);
        self.factors.push(value);
    }
    /// Extend the product with another one, without normalizing it
    pub fn extend(&mut self, other: Mul<T>) {
        self.set_normalized(false);
        self.factors.extend(other.factors);
    }
    /// Return an iterator over the product factors.
    pub fn iter(&self) -> std::slice::Iter<'_, Term<T>> {
        self.factors.iter()
    }
    /// Checks if the product have a constant coefficient.
    pub fn has_coeff(&self) -> bool {
        if let Term::Value(_) = &self.factors.last().unwrap() {
            true
        } else {
            false
        }
    }
    /// Set the constant coefficient factor of the product
    pub fn set_coeff(&mut self, coeff: T::Element) {
        if let Some(Term::Value(v)) = self.factors.last_mut() {
            v.value = coeff
        } else {
            self.factors.push(Term::Value(Value::new(coeff, self.ring)));
        }
    }
}

impl<T: RingBound> Mul<T> {
    /// Return the constant coefficient factor of the product if it exists, one otherwise.
    pub fn get_coeff(&self) -> T::Element {
        if let Term::Value(v) = &self.factors.last().unwrap() {
            v.value.clone()
        } else {
            self.ring.one()
        }
    }
}

impl<T: RingBound> IntoIterator for Mul<T> {
    type Item = Term<T>;

    type IntoIter = std::vec::IntoIter<Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.into_iter()
    }
}

impl<'a, T: RingBound> IntoIterator for &'a Mul<T> {
    type Item = &'a Term<T>;

    type IntoIter = std::slice::Iter<'a, Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.iter()
    }
}

impl<T: RingBound> Print for Mul<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print the cefficient at the beginning of the product
        if self.has_coeff() {
            Self::group(self.factors.last().unwrap(), options, f)?;
        }
        for (i, factor) in self.factors.iter().enumerate() {
            if let Term::Value(_) = factor {
                continue;
            }
            Self::group(factor, options, f)?;
            if i != self.len() - 1 {
                Self::operator("*", options, f)?;
            }
        }
        Ok(())
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        let mut den: Option<PrettyPrinter> = None;
        let mut num: Option<PrettyPrinter> = None;
        // Print the cefficient at the beginning of the product
        if self.has_coeff() {
            num = Some(Print::pretty_print(self.factors.last().unwrap(), options))
        }
        for factor in self.factors.iter() {
            if let Term::Value(_) = factor {
                continue;
            }
            let printed = match factor {
                Term::Value(_) => continue,
                Term::Pow(pow) if pow.exposant.is_strictly_negative() => {
                    let mut factor = pow.clone();
                    **factor.exposant = -&**factor.exposant;
                    &Term::Pow(factor).normalize()
                }
                _ => factor,
            };
            let mut elem = Print::pretty_print(printed, options);
            if match printed {
                Term::Add(_) => true,
                _ => false,
            } {
                elem.paren();
            }
            match factor {
                Term::Pow(pow) if pow.exposant.is_strictly_negative() => {
                    if let Some(den) = &mut den {
                        den.concat("⋅", false, &elem)
                    } else {
                        den = Some(elem);
                    }
                }
                _ => {
                    if let Some(num) = &mut num {
                        num.concat("⋅", false, &elem)
                    } else {
                        num = Some(elem);
                    }
                }
            };
        }

        let mut num = if let Some(num) = num {
            num
        } else {
            Print::pretty_print(&Term::one(self.ring), options)
        };
        if let Some(den) = den {
            num.vertical_concat("─", &den);
        }
        num
    }
}

impl<T: RingBound> Display for Mul<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
