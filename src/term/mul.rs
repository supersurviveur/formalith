//! Product expression.

use std::fmt::Display;

use crate::{
    field::{Group, Ring, Set},
    printer::{PrettyPrint, PrettyPrinter, Print, PrintOptions},
    term::{Normalize, Value},
};

use super::{Flags, Term};

/// A product of expressions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mul<T: Set> {
    flags: u8,
    pub(crate) coefficient: <T::ProductCoefficientSet as Set>::Element,
    pub(crate) factors: Vec<Term<T>>,
    pub(crate) set: T,
}

impl<T: Set> Flags for Mul<T> {
    fn get_flags(&self) -> u8 {
        self.flags
    }
    fn get_flags_mut(&mut self) -> &mut u8 {
        &mut self.flags
    }
}

impl<T: Set> Mul<T> {
    /// Create a new product expression
    pub fn new(
        coefficient: <T::ProductCoefficientSet as Set>::Element,
        factors: Vec<Term<T>>,
        ring: T,
    ) -> Self {
        Self {
            flags: 0,
            coefficient,
            factors,
            set: ring,
        }
    }
    /// Create a new product expression
    pub fn empty(ring: T) -> Self {
        Self::new(ring.get_coefficient_set().nth(1), vec![], ring)
    }
    /// Create a new empty product, with an inner Vec with a defined capacity.
    pub fn with_capacity(capacity: usize, ring: T) -> Self {
        Self {
            flags: 0,
            coefficient: ring.get_coefficient_set().nth(1),
            factors: Vec::with_capacity(capacity),
            set: ring,
        }
    }
    /// Returns true if the multiplication is empty.
    pub fn is_empty(&self) -> bool {
        self.factors.is_empty()
    }
    /// Get the number of terms in the sum.
    pub fn len(&self) -> usize {
        self.factors.len() + 1
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
    /// Checks if the product have a constant coefficient not equal to one.
    pub fn has_coeff(&self) -> bool {
        !self.set.get_coefficient_set().is_one(&self.coefficient)
    }
    /// Set the constant coefficient factor of the product
    pub fn set_coeff(&mut self, coeff: <T::ProductCoefficientSet as Set>::Element) {
        self.coefficient = coeff;
    }
}

impl<T: Set> Mul<T> {
    /// Return the constant coefficient factor of the product.
    pub fn get_coeff(&self) -> <T::ProductCoefficientSet as Set>::Element {
        self.coefficient.clone()
    }
    /// Return the constant coefficient factor of the product through a mutable reference.
    pub fn get_coeff_mut(&mut self) -> &mut <T::ProductCoefficientSet as Set>::Element {
        &mut self.coefficient
    }
}

impl<T: Set> IntoIterator for Mul<T> {
    type Item = Term<T>;

    type IntoIter = std::vec::IntoIter<Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.into_iter()
    }
}

impl<'a, T: Set> IntoIterator for &'a Mul<T> {
    type Item = &'a Term<T>;

    type IntoIter = std::slice::Iter<'a, Term<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.factors.iter()
    }
}

impl<T: Set> Print for Mul<T> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print the cefficient at the beginning of the product
        if self.has_coeff() {
            self.set
                .get_coefficient_set()
                .print(&self.coefficient, options, f)?;
        }
        for (i, factor) in self.factors.iter().enumerate() {
            Self::group(factor, options, f)?;
            if i != self.len() - 1 {
                Self::operator("*", options, f)?;
            }
        }
        Ok(())
    }
}
impl<T: Set> Mul<T> {
    fn pretty_print_default<F: Fn() -> crate::printer::PrettyPrinter>(
        &self,
        f: F,
        options: &PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        let mut den: Option<PrettyPrinter> = None;
        let mut num: Option<PrettyPrinter> = None;
        // Print the cefficient at the beginning of the product
        if self.has_coeff() {
            num = Some(PrettyPrint::pretty_print(
                &Term::Value(Value::new(
                    self.coefficient.clone(),
                    self.set.get_coefficient_set(),
                )),
                options,
            ))
        }
        for factor in self.factors.iter() {
            let printed = match factor {
                Term::Pow(pow) if pow.exponant.is_strictly_negative() => {
                    let mut factor = pow.clone();
                    **factor.exponant = -&**factor.exponant;
                    &Term::Pow(factor).normalize()
                }
                _ => factor,
            };
            let mut elem = PrettyPrint::pretty_print(printed, options);
            if matches!(printed, Term::Add(_)) {
                elem.paren();
            }
            match factor {
                Term::Pow(pow) if pow.exponant.is_strictly_negative() => {
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

        let mut num = num.unwrap_or_else(f);
        if let Some(den) = den {
            num.vertical_concat("─", &den);
        }
        num
    }
}
impl<T: Set> PrettyPrint for Mul<T> {
    default fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        self.pretty_print_default(|| unreachable!("Pretty printing a `Mul` in a set or a group should return a numerator. This is a bug, this product is not normalized or represents an impossible product (not living in a ring)"), options)
    }
}

impl<T: Ring> PrettyPrint for Mul<T> {
    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        self.pretty_print_default(
            || PrettyPrint::pretty_print(&Term::one(self.set), options),
            options,
        )
    }
}

impl<T: Set> Display for Mul<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        PrettyPrint::fmt(self, &PrintOptions::default(), f)
    }
}
