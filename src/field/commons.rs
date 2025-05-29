//! Most commons groups, rings and fields like [R], [C] or [M].
//! These are currently only implemented using malachite arbitrary precision numbers.

use malachite::base::num::arithmetic;
use malachite::base::num::arithmetic::traits::CheckedRoot;
use malachite::base::num::arithmetic::traits::Parity;
use malachite::base::num::arithmetic::traits::Sign;
use malachite::base::num::basic::traits::NegativeOne;
use malachite::base::num::basic::traits::One;
use malachite::base::num::basic::traits::Zero;
use malachite::base::num::{
    arithmetic::traits::Pow, basic::traits, conversion::traits::FromSciString,
};
use malachite::rational::Rational;
use malachite::Integer;
use malachite::Natural;
use std::cell::OnceCell;
use std::cell::RefCell;
use std::error::Error;
use std::fmt::Display;
use std::mem::MaybeUninit;
use std::{cmp::Ordering, fmt, marker::PhantomData};

use crate::combinatorics;
use crate::context::Context;
use crate::matrix::Matrix;
use crate::parser::lexer::TokenKind;
use crate::parser::parser::Parser;
use crate::parser::parser::ParserError;
use crate::parser::parser::ParserTrait;
use crate::printer::PrettyPrinter;
use crate::printer::Print;
use crate::term;
use crate::term::Mul;
use crate::term::TermField;
use crate::{
    context::Symbol,
    term::{Term, Value},
};

use super::GroupImpl;
use super::RingImpl;
use super::TryElementCast;
use super::TryExprCast;
use super::{Derivable, Field, Group, Ring};

/// The integer ring
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Z<T> {
    phantom: PhantomData<T>,
}

impl<T: Clone> Copy for Z<T> {}

impl GroupImpl for Z<Integer> {
    type Element = Integer;

    type ExposantSet = Z<malachite::Integer>; // TODO change type

    fn get_exposant_set(&self) -> Self::ExposantSet {
        *self
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

impl RingImpl for Z<Integer> {
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
pub const Z: Z<Integer> = Z {
    phantom: PhantomData,
};

/// The real field.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct R<T> {
    phantom: PhantomData<T>,
}

impl<T: Clone> Copy for R<T> {}

impl GroupImpl for R<malachite::rational::Rational> {
    type Element = Rational;

    type ExposantSet = R<Rational>;

    fn get_exposant_set(&self) -> Self::ExposantSet {
        *self
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

impl RingImpl for R<Rational> {
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

    fn as_fraction(&self, a: &Self::Element) -> (Self::Element, Self::Element) {
        let sign = a.sign();
        let (num, den) = a.to_numerator_and_denominator();
        let (mut num, den) = (num.into(), den.into());
        if sign == Ordering::Less {
            num *= -Rational::ONE;
        }
        (num, den)
    }

    fn normalize(&self, a: Term<Self>) -> Term<Self> {
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
                                *self,
                            )),
                            (**base.exposant).clone(),
                        ],
                        self.get_exposant_set(),
                    );

                    let mul = Term::Mul(mul).normalize();
                    let mut res = if mul.is_one() {
                        *res.base.clone()
                    } else if mul.is_zero() {
                        Term::Value(Value::new(<Rational as traits::One>::ONE, *self))
                    } else {
                        *res.exposant = mul.into();
                        Term::Pow(res)
                    };
                    if denominator != 1 {
                        res = term::Pow::new(
                            res,
                            Term::Value(Value::new(
                                Rational::from_naturals(Natural::ONE, denominator),
                                *self,
                            )),
                            *self,
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
                                return Term::Value(Value::new(res, *self));
                            }
                        }
                    }
                    return Term::Pow(term::Pow::new(
                        Term::Value(Value::new(res, *self)),
                        Term::Value(Value::new(Rational::from_naturals(num, den), *self)),
                        *self,
                    ));
                }
                _ => a,
            },
            Term::Fun(fun) => match fun {
                // Context::ABS => {
                //     // TODO check parameters count, maybe at function creation in parser
                //     let content = fun.args.first().unwrap();
                //     if content >= &0.into() {
                //         content.clone()
                //     } else if content < &0.into() {
                //         -content
                //     } else {
                //         a
                //     }
                // }
                _ => a,
            },
            _ => a,
        }
    }

    fn expand(&self, a: Term<Self>) -> Term<Self> {
        match &a {
            Term::Pow(term::Pow { base, exposant, .. }) => {
                if let (Term::Add(add), Term::Value(value)) = (&**base, &***exposant) {
                    if value.value.denominator_ref() == &Natural::ONE
                        && value.value.numerator_ref() != &Natural::ONE
                    {
                        if let Ok(num) = usize::try_from(value.value.numerator_ref()) {
                            let mut terms = Term::zero(*self);

                            let mut iterator =
                                combinatorics::CompositionIterator::new(add.len(), num);

                            while let Some(k) = iterator.next() {
                                let ik = k
                                    .iter()
                                    .map(|x| Integer::from(*x))
                                    .collect::<Vec<Integer>>();
                                let mut result = Term::Value(Value::new(
                                    combinatorics::Combinatorics::multinom(&ik).into(),
                                    *self,
                                ));

                                for (i, term) in add.iter().enumerate() {
                                    result *= term.pow(
                                        &Value::new(Rational::from_unsigneds(k[i], 1), *self)
                                            .into(),
                                    );
                                }

                                terms += result;
                            }
                            if value.value.sign() == Ordering::Less {
                                return term::Pow::new(
                                    terms,
                                    Term::Value(Value::new(Rational::NEGATIVE_ONE, *self)),
                                    *self,
                                )
                                .into();
                            } else {
                                return terms;
                            }
                        }
                    }
                } else if let (Term::Mul(mul), _) = (&**base, &**exposant) {
                    return Mul::new(
                        mul.into_iter()
                            .map(|x| term::Pow::new(x.clone(), (**exposant).clone(), *self).into())
                            .collect(),
                        *self,
                    )
                    .into();
                }
                a
            }
            _ => a,
        }
    }
}

impl Field for R<Rational> {}

impl Derivable for R<Rational> {
    fn derivative(&self, _: &Self::Element, _: Symbol) -> Self::Element {
        Rational::ZERO
    }
}

/// The real field, using [malachite::Rational] as constants to get arbitrary precision numbers.
pub const R: R<Rational> = R {
    phantom: PhantomData,
};

/// An enum representing elements inside a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorSpaceElement<T: Group, Vector> {
    /// A scalar element living in [T]
    Scalar(T::Element),
    /// A vector living in the vector space
    Vector(Vector),
}

impl<T: Group, Vector: PartialEq> PartialOrd for VectorSpaceElement<T, Vector>
where
    T::Element: PartialOrd,
{
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

/// The matrix ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct M<T: Ring> {
    pub(crate) scalar_sub_set: T,
}

impl<T: Ring> M<T> {
    pub fn new(scalar_sub_set: T) -> Self {
        Self { scalar_sub_set }
    }
}

impl<T: Ring> GroupImpl for M<T> {
    type Element = VectorSpaceElement<T, Matrix<TermField<T>>>;

    type ExposantSet = Self;

    fn get_exposant_set(&self) -> Self::ExposantSet {
        *self
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
    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String> {
        Ok(VectorSpaceElement::Scalar(
            self.scalar_sub_set.parse_litteral(value)?,
        ))
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

impl<T: Ring> RingImpl for M<T> {
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
        match a {
            VectorSpaceElement::Scalar(scalar) => self
                .scalar_sub_set
                .try_inv(scalar)
                .map(|s| VectorSpaceElement::Scalar(s)),
            VectorSpaceElement::Vector(matrix) => {
                matrix.inv().ok().map(|m| VectorSpaceElement::Vector(m))
            }
        }
    }

    fn as_fraction(&self, a: &Self::Element) -> (Self::Element, Self::Element) {
        let num = match &a {
            VectorSpaceElement::Vector(matrix) => {
                let mut res = matrix.clone();
                res.data.iter_mut().for_each(|x| {
                    *x = x.unify();
                });
                VectorSpaceElement::Vector(res)
            }
            _ => a.clone(),
        };
        (num, self.one())
    }

    fn normalize(&self, a: Term<Self>) -> Term<Self> {
        // Try downcasting the expression and apply normalization over the scalar set
        match <T as TryExprCast<Self>>::downcast_expr(&self.scalar_sub_set, a.clone()) {
            Ok(expr) => {
                return <T as TryExprCast<Self>>::upcast_expr(
                    *self,
                    self.scalar_sub_set.normalize(expr),
                )
                .expect("Converting back to matrix should always be possible")
            }
            Err(_) => {}
        }
        match &a {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    res.data.iter_mut().for_each(|x| *x = x.normalize());
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                VectorSpaceElement::Scalar(_) => a,
            },
            Term::Pow(term::Pow { base, exposant, .. }) => match (&**base, &***exposant) {
                (
                    Term::Value(Value {
                        value: VectorSpaceElement::Vector(matrix),
                        ..
                    }),
                    Term::Value(Value {
                        value: exposant, ..
                    }),
                ) => {
                    if self.partial_cmp(exposant, &self.zero()) == Some(Ordering::Less) {
                        Term::Pow(term::Pow::new(
                            Term::Value(Value::new(
                                VectorSpaceElement::Vector(
                                    matrix
                                        .inv()
                                        .expect(&format!("Cannot invert matrix {}", matrix)),
                                ),
                                *self,
                            )),
                            Term::Value(Value::new(self.neg(exposant), *self)),
                            *self,
                        ))
                    } else {
                        a
                    }
                }
                _ => a,
            },
            _ => a,
        }
    }

    fn expand(&self, a: Term<Self>) -> Term<Self> {
        match &a {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    res.data.iter_mut().for_each(|x| {
                        *x = x.expand();
                    });
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                VectorSpaceElement::Scalar(_) => a,
            },
            _ => a,
        }
    }

    fn simplify(&self, a: Term<Self>) -> Term<Self> {
        match &a {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    res.data.iter_mut().for_each(|x| {
                        *x = x.simplify();
                    });
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                _ => a,
            },
            _ => a,
        }
    }
    fn parse_expression(&self, parser: &mut Parser) -> Result<Option<Term<Self>>, ParserError> {
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
                        self.scalar_sub_set.get_term_field(),
                    )),
                    *self,
                ))))
            }
            _ => Ok(None),
        }
    }
}

/// Failed to convert a term into another term.
#[derive(Debug, Clone, Copy)]
pub struct TryCastError();
impl Error for TryCastError {}

impl fmt::Display for TryCastError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Can't cast to this group")
    }
}

impl<T: Ring> TryElementCast<M<T>> for T {
    fn downcast_element(
        value: <M<T> as GroupImpl>::Element,
    ) -> Result<Self::Element, TryCastError> {
        match value {
            VectorSpaceElement::Scalar(scalar) => Ok(scalar),
            VectorSpaceElement::Vector(_) => Err(TryCastError()),
        }
    }

    fn upcast_element(value: Self::Element) -> Result<<M<T> as GroupImpl>::Element, TryCastError> {
        Ok(VectorSpaceElement::Scalar(value))
    }
}

impl<T: Group> TryElementCast<T> for T {
    fn downcast_element(value: <T>::Element) -> Result<Self::Element, TryCastError> {
        Ok(value)
    }

    fn upcast_element(value: Self::Element) -> Result<<T>::Element, TryCastError> {
        Ok(value)
    }
}

/// The matrix field, with real coefficients. See [R]
pub const M: M<R<Rational>> = M { scalar_sub_set: R };

impl From<usize> for Term<R<Rational>> {
    fn from(value: usize) -> Self {
        Term::Value(Value::new(Rational::from(value), R))
    }
}

impl<T> TryFrom<Term<R<T>>> for <R<T> as GroupImpl>::Element
where
    R<T>: Ring,
{
    type Error = &'static str;

    fn try_from(value: Term<R<T>>) -> Result<Self, Self::Error> {
        match value {
            Term::Value(value) => Ok(value.get_value()),
            _ => Err("Value is not a constant"),
        }
    }
}
