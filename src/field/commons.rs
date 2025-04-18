//! Most commons groups, rings and fields like [R], [C] or [M].
//! These are currently only implemented using malachite arbitrary precision numbers.

use malachite::base::num::arithmetic;
use malachite::base::num::arithmetic::traits::CheckedRoot;
use malachite::base::num::arithmetic::traits::Parity;
use malachite::base::num::arithmetic::traits::Sign;
use malachite::base::num::basic::traits::One as MalachiteOne;
use malachite::base::num::basic::traits::Zero;
use malachite::base::num::{
    arithmetic::traits::Pow, basic::traits, conversion::traits::FromSciString,
};
use malachite::rational::Rational;
use malachite::Integer;
use malachite::Natural;
use std::fmt::Display;
use std::{cmp::Ordering, fmt, marker::PhantomData};

use crate::context::Context;
use crate::matrix::Matrix;
use crate::parser::lexer::TokenKind;
use crate::parser::parser::Parser;
use crate::parser::parser::ParserError;
use crate::printer::PrettyPrinter;
use crate::printer::Print;
use crate::term;
use crate::term::TermField;
use crate::{
    context::Symbol,
    term::{Term, Value},
};

use super::{Derivable, Field, Group, Ring};

/// The integer ring
#[derive(Clone, Debug, PartialEq)]
pub struct Z<T> {
    phantom: PhantomData<T>,
}

impl Group for Z<Integer> {
    type Element = Integer;

    type ExposantSet = Z<malachite::Integer>; // TODO change type

    fn get_exposant_set(&self) -> &Self::ExposantSet {
        &Z
    }

    fn zero(&self) -> Self::Element {
        Integer::ZERO
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }

    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        a.partial_cmp(b)
    }

    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String> {
        Integer::from_sci_string(value).ok_or(format!(
            "Failed to parse \"{}\" to malachite::Integer",
            value
        ))
    }

    fn pretty_print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        todo!()
    }
}

impl Ring for Z<Integer> {
    fn one(&self) -> Self::Element {
        <Integer as malachite::base::num::basic::traits::One>::ONE
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn nth(&self, nth: i64) -> Self::Element {
        Integer::const_from_signed(nth)
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

/// The integer ring, using [malachite::Integer] as constant to get arbitrary precision integers.
pub const Z: &Z<Integer> = &Z {
    phantom: PhantomData,
};

/// The real field.
#[derive(Clone, Debug, PartialEq)]
pub struct R<T> {
    phantom: PhantomData<T>,
}

impl Group for R<malachite::rational::Rational> {
    type Element = Rational;

    type ExposantSet = R<Rational>;

    fn get_exposant_set(&self) -> &Self::ExposantSet {
        self
    }

    fn zero(&self) -> Self::Element {
        malachite::rational::Rational::ZERO
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }

    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        a.partial_cmp(b)
    }

    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String> {
        Rational::from_sci_string(value).ok_or(format!(
            "Failed to parse \"{}\" to malachite::Rational",
            value
        ))
    }

    fn normalize(&'static self, a: Term<Self>) -> Term<Self> {
        match &a {
            Term::Pow(term::Pow { base, exposant, .. }) => match (&**base, &***exposant) {
                (Term::Pow(base), Term::Value(exposant)) => {
                    let (numerator, denominator) = exposant.value.to_numerator_and_denominator();
                    let mut res = base.clone();

                    // Numerator and sign can always be passed to the product
                    // If denominator is odd, we can pass it to the product too
                    let (passed_denominator, denominator) = if denominator.odd() {
                        (denominator, Natural::ONE)
                    } else {
                        (Natural::ONE, denominator)
                    };
                    let mul = term::Mul::new(
                        vec![
                            Term::Value(Value::new(
                                Rational::from_sign_and_naturals(
                                    exposant.value.sign() == Ordering::Greater,
                                    numerator.clone(),
                                    passed_denominator,
                                ),
                                self,
                            )),
                            (**base.exposant).clone(),
                        ],
                        self.get_exposant_set(),
                    );

                    let mul = Term::Mul(mul).normalize();
                    let mut res = if mul.is_one() {
                        *res.base.clone()
                    } else if mul.is_zero() {
                        Term::Value(Value::new(<Rational as traits::One>::ONE, self))
                    } else {
                        *res.exposant = mul.into();
                        Term::Pow(res)
                    };
                    if denominator != 1 {
                        res = term::Pow::new(
                            res,
                            Term::Value(Value::new(
                                Rational::from_naturals(Natural::ONE, denominator),
                                self,
                            )),
                            self,
                        )
                        .into();
                    }
                    res
                }
                (Term::Value(base), Term::Value(exposant)) => {
                    let exposant = &exposant.value;
                    let mut res = base.value.clone();
                    if exposant.sign() == Ordering::Less {
                        res = Pow::pow(res, -1_i64);
                    }
                    let (mut num, den) = exposant.to_numerator_and_denominator();
                    if let Ok(exposant) = i64::try_from(&num) {
                        res = arithmetic::traits::Pow::pow(res, exposant);
                        num = Natural::ONE;
                        if let Ok(root_exposant) = i64::try_from(&den) {
                            if let Some(root_free) = (&res).checked_root(root_exposant) {
                                res = root_free;
                                return Term::Value(Value::new(res, self));
                            }
                        }
                    }
                    return Term::Pow(term::Pow::new(
                        Term::Value(Value::new(res, self)),
                        Term::Value(Value::new(Rational::from_naturals(num, den), self)),
                        self,
                    ));
                }
                _ => a,
            },
            Term::Fun(fun) => match fun.ident {
                Context::ABS => {
                    // TODO check parameters count, maybe at function creation in parser
                    let content = fun.args.first().unwrap();
                    if content >= &0.into() {
                        content.clone()
                    } else if content < &0.into() {
                        -content
                    } else {
                        a
                    }
                }
                _ => a,
            },
            _ => a,
        }
    }

    fn pretty_print(
        &self,
        elem: &Self::Element,
        _options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        let (num, den) = elem.numerator_and_denominator_ref();
        let mut num = PrettyPrinter::from(if elem.sign() == Ordering::Less {
            format!("-{}", num)
        } else {
            format!("{}", num)
        });
        if *den == 1 {
            num
        } else {
            let den = PrettyPrinter::from(format!("{}", den));
            num.vertical_concat("─", &den);
            num
        }
    }
}

impl Ring for R<Rational> {
    fn one(&self) -> Self::Element {
        <Rational as malachite::base::num::basic::traits::One>::ONE
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn nth(&self, nth: i64) -> Self::Element {
        Rational::const_from_signed(nth)
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        Some(a.pow(-1_i64))
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        *a == self.one()
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(a, b);
    }
}

impl Field for R<Rational> {}

impl Derivable for R<Rational> {
    fn derivative(&self, _: &Self::Element, _: Symbol) -> Self::Element {
        Rational::ZERO
    }
}

/// The real field, using [malachite::Rational] as constants to get arbitrary precision numbers.
pub const R: &R<Rational> = &R {
    phantom: PhantomData,
};

/// An enum representing elements inside a vector space.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorSpaceElement<T: Group, Vector> {
    /// A scalar element living in [T]
    Scalar(T::Element),
    /// A vector living in the vector space
    Vector(Vector),
}

impl<T: Group, Vector: PartialEq> PartialOrd for VectorSpaceElement<T, Vector> {
    #[inline]
    fn partial_cmp(&self, other: &VectorSpaceElement<T, Vector>) -> Option<Ordering> {
        match (self, other) {
            (VectorSpaceElement::Scalar(s1), VectorSpaceElement::Scalar(s2)) => {
                PartialOrd::partial_cmp(s1, s2)
            }
            _ => None,
        }
    }
}

impl<T: Group, Vector: Display> Display for VectorSpaceElement<T, Vector> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorSpaceElement::Scalar(scalar) => write!(f, "{}", scalar),
            VectorSpaceElement::Vector(vector) => write!(f, "{}", vector),
        }
    }
}

/// The matrix field.
#[derive(Debug, Clone, PartialEq)]
pub struct M<T: Group> {
    scalar_sub_set: &'static T,
    term_sub_set: &'static TermField<T>,
}

impl<T: Ring> Group for M<T> {
    type Element = VectorSpaceElement<T, Matrix<TermField<T>>>;

    type ExposantSet = T;

    fn get_exposant_set(&self) -> &Self::ExposantSet {
        self.scalar_sub_set
    }

    fn zero(&self) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.zero())
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (VectorSpaceElement::Scalar(a), VectorSpaceElement::Scalar(b)) => {
                VectorSpaceElement::Scalar(self.scalar_sub_set.add(a, b))
            }
            (VectorSpaceElement::Vector(a), VectorSpaceElement::Vector(b)) => {
                VectorSpaceElement::Vector(a + b)
            }
            _ => panic!("Can't add a scalar and a vector ! ({} + {})", a, b),
        }
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        match a {
            VectorSpaceElement::Scalar(a) => VectorSpaceElement::Scalar(self.scalar_sub_set.neg(a)),
            VectorSpaceElement::Vector(a) => VectorSpaceElement::Vector(-a),
        }
    }

    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        match (a, b) {
            (VectorSpaceElement::Scalar(a), VectorSpaceElement::Scalar(b)) => {
                self.scalar_sub_set.partial_cmp(a, b)
            }
            _ => None,
        }
    }

    fn normalize(&'static self, a: Term<Self>) -> Term<Self> {
        match a {
            Term::Value(value) => match value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    for (i, v) in matrix.data.into_iter().enumerate() {
                        res.data[i] = self.scalar_sub_set.normalize(v);
                    }
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), self))
                }
                VectorSpaceElement::Scalar(_) => Term::Value(value),
            },
            _ => a,
        }
    }

    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String> {
        Ok(VectorSpaceElement::Scalar(
            self.scalar_sub_set.parse_litteral(value)?,
        ))
    }
    fn parse_expression(
        &'static self,
        parser: &mut Parser,
    ) -> Result<Option<Term<Self>>, ParserError> {
        match parser.token.kind {
            TokenKind::OpenBracket => {
                parser.next_token();
                let mut lines = vec![];
                let mut size = None;
                loop {
                    parser.expect_token(TokenKind::OpenBracket)?;

                    let mut line_size = 0;
                    loop {
                        lines.push(parser.parse_expression(0, self.scalar_sub_set)?.unwrap());
                        line_size += 1;

                        if parser.token.kind != TokenKind::Comma {
                            break;
                        }
                        parser.next_token();
                    }
                    if let Some(size) = size {
                        if size != line_size {
                            return Err(ParserError::new(
                                "Matrix dimension aren't coherent !".to_string(),
                            ));
                        }
                    } else {
                        size = Some(line_size)
                    }

                    parser.expect_token(TokenKind::CloseBracket)?;

                    if parser.token.kind != TokenKind::Comma {
                        break;
                    }
                    parser.next_token();
                }
                parser.expect_token(TokenKind::CloseBracket)?;
                Ok(Some(Term::Value(Value::new(
                    VectorSpaceElement::Vector(Matrix::new(
                        (lines.len() / size.unwrap(), size.unwrap()),
                        lines,
                        self.term_sub_set,
                    )),
                    self,
                ))))
            }
            _ => Ok(None),
        }
    }

    fn pretty_print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        match elem {
            VectorSpaceElement::Scalar(scalar) => self.scalar_sub_set.pretty_print(scalar, options),
            VectorSpaceElement::Vector(matrix) => matrix.pretty_print(options),
        }
    }
}

impl<T: Ring> Ring for M<T> {
    fn one(&self) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.one())
    }

    fn nth(&self, nth: i64) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.nth(nth))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        todo!()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        todo!()
    }
}

/// The matrix field, with real coefficients. See [R]
pub const M: &M<R<Rational>> = &M {
    scalar_sub_set: R,
    term_sub_set: &TermField::new(R),
};

impl From<usize> for Term<R<Rational>> {
    fn from(value: usize) -> Self {
        Term::Value(Value::new(Rational::from(value), R))
    }
}
