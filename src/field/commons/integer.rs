//! The integer ring `{$ZZ$}`

use std::{cmp::Ordering, marker::PhantomData};

use malachite::{
    Integer,
    base::num::{
        basic::traits::{One, Zero},
        conversion::traits::FromSciString,
    },
};

use crate::{
    field::{Group, PartiallyOrderedSet, Ring, Set, SetParseExpression},
    parser::Parser,
    printer::PrettyPrinter,
};

/// The integer ring `{$ZZ$}`
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Z<T> {
    phantom: PhantomData<T>,
}

impl<T: Clone> Copy for Z<T> {}

impl Set for Z<Integer> {
    type Element = Integer;

    type ExponantSet = Self; // TODO change type to N ?
    type ProductCoefficientSet = Self;

    fn get_exponant_set(&self) -> Self::ExponantSet {
        *self
    }
    fn get_coefficient_set(&self) -> Self::ProductCoefficientSet {
        *self
    }
    fn print(
        &self,
        elem: &Self::Element,
        _: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{elem}")
    }
    fn pretty_print(
        &self,
        elem: &Self::Element,
        _: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        PrettyPrinter::from(format!("{elem}"))
    }

    fn element_eq(&self, a: &Self::Element, b: &Self::Element) -> bool {
        a == b
    }
}

impl PartiallyOrderedSet for Z<Integer> {
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        a.partial_cmp(b)
    }
}

impl Group for Z<Integer> {
    fn zero(&self) -> Self::Element {
        Integer::ZERO
    }

    fn nth(&self, nth: i64) -> <Self::ProductCoefficientSet as Set>::Element {
        Integer::const_from_signed(nth)
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }
}

impl Ring for Z<Integer> {
    fn one(&self) -> Self::Element {
        Integer::ONE
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if *a == 1 {
            Some(Integer::const_from_unsigned(1))
        } else if *a == -1 {
            Some(Integer::const_from_signed(-1))
        } else {
            None
        }
    }
}
impl<N> SetParseExpression<N> for Z<Integer> {
    fn parse_literal(&self, parser: &mut Parser) -> Result<Option<Self::Element>, String> {
        parser.is_literal_and(|value| {
            Ok(Some(Integer::from_sci_string(value).ok_or_else(|| {
                format!("Failed to parse \"{value}\" to malachite::Integer",)
            })?))
        })
    }
}

/// The integer ring, using [`malachite::Integer`] as constant to get arbitrary precision integers.
pub const Z: Z<Integer> = Z {
    phantom: PhantomData,
};
