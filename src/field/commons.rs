//! Most commons groups, rings and fields like [struct@R] or [struct@M].
//! These are currently only implemented using malachite arbitrary precision numbers.

use malachite::Integer;
use malachite::Natural;
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
use std::error::Error;
use std::fmt::Display;
use std::{cmp::Ordering, fmt, marker::PhantomData};

use crate::combinatorics;
use crate::context::Context;
use crate::matrix::Matrix;
use crate::matrix::MatrixResult;
use crate::parser::Parser;
use crate::parser::ParserError;
use crate::parser::ParserTrait;
use crate::parser::lexer::TokenKind;
use crate::printer::PrettyPrinter;
use crate::printer::Print;
use crate::term;
use crate::term::Fun;
use crate::term::Mul;
use crate::term::TermField;
use crate::{
    context::Symbol,
    term::{Term, Value},
};

use super::Group;
use super::Ring;
use super::TryElementFrom;
use super::{Derivable, Field, GroupBound, RingBound};

/// The integer ring
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Z<T> {
    phantom: PhantomData<T>,
}

impl<T: Clone> Copy for Z<T> {}

impl Group for Z<Integer> {
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
        Integer::from_sci_string(value)
            .ok_or(format!("Failed to parse \"{value}\" to malachite::Integer",))
    }
    fn print(
        &self,
        elem: &Self::Element,
        _: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", elem)
    }
    fn pretty_print(
        &self,
        elem: &Self::Element,
        _: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        PrettyPrinter::from(format!("{elem}"))
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
pub const Z: Z<Integer> = Z {
    phantom: PhantomData,
};

/// The real field.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct R<T> {
    phantom: PhantomData<T>,
}

impl<T: Clone> Copy for R<T> {}

impl Group for R<malachite::rational::Rational> {
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
            "Failed to parse \"{value}\" to malachite::Rational",
        ))
    }
    fn print(
        &self,
        elem: &Self::Element,
        _: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", elem)
    }
    fn pretty_print(
        &self,
        elem: &Self::Element,
        _options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        let (num, den) = elem.numerator_and_denominator_ref();
        let mut num = PrettyPrinter::from(if elem.sign() == Ordering::Less {
            format!("-{num}")
        } else {
            format!("{num}")
        });
        if *den == 1 {
            num
        } else {
            let den = PrettyPrinter::from(format!("{den}"));
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
                    return res;
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
                        if let Ok(root_exposant) = i64::try_from(&den)
                            && let Some(root_free) = (&res).checked_root(root_exposant)
                        {
                            res = root_free;
                            return Term::Value(Value::new(res, *self));
                        }
                    }
                    return Term::Pow(term::Pow::new(
                        Term::Value(Value::new(res, *self)),
                        Term::Value(Value::new(Rational::from_naturals(num, den), *self)),
                        *self,
                    ));
                }
                _ => {}
            },
            Term::Fun(fun) => match fun.get_ident() {
                Context::ABS => {
                    // TODO check parameters count, maybe at function creation in parser
                    if let Some(fun) = fun.as_any().downcast_ref::<Fun<Self>>() {
                        let content = fun.args.first().unwrap();
                        if content >= &0.into() {
                            return content.clone();
                        } else if content < &0.into() {
                            return -content;
                        }
                    }
                }
                Context::DET => {
                    if let Some(Term::Value(matrix)) = fun.get_arg::<M<Self>>()
                        && let Ok(res) = matrix.value.det(*self)
                    {
                        return res;
                    }
                }
                _ => {}
            },
            _ => {}
        }
        a
    }

    fn expand(&self, a: Term<Self>) -> Term<Self> {
        match &a {
            Term::Pow(term::Pow { base, exposant, .. }) => {
                if let (Term::Add(add), Term::Value(value)) = (&**base, &***exposant) {
                    if value.value.denominator_ref() == &Natural::ONE
                        && value.value.numerator_ref() != &Natural::ONE
                        && let Ok(num) = usize::try_from(value.value.numerator_ref())
                    {
                        let mut terms = Term::zero(*self);

                        let mut iterator = combinatorics::CompositionIterator::new(add.len(), num);

                        while let Some(k) = iterator.next_composition() {
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
                                    &Value::new(Rational::from_unsigneds(k[i], 1), *self).into(),
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

/// The real field, using [malachite::rational::Rational] as constants to get arbitrary precision numbers.
pub const R: R<Rational> = R {
    phantom: PhantomData,
};

/// An enum representing elements inside a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorSpaceElement<T: GroupBound, Vector> {
    /// A scalar element living in `T`
    Scalar(T::Element),
    /// A vector living in the vector space
    Vector(Vector),
}

impl<T: RingBound> VectorSpaceElement<TermField<T>, Matrix<TermField<T>>> {
    /// Compute the determinant of self in set `T`.
    pub fn det(&self, _set: T) -> MatrixResult<Term<T>> {
        match self {
            VectorSpaceElement::Scalar(scalar) => Ok(scalar.clone()),
            VectorSpaceElement::Vector(matrix) => matrix.det(),
        }
    }
}

impl<T: GroupBound, Vector: PartialEq> PartialOrd for VectorSpaceElement<T, Vector>
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

impl<T: GroupBound, Vector: Display> Display for VectorSpaceElement<T, Vector> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VectorSpaceElement::Scalar(scalar) => write!(f, "{}", scalar),
            VectorSpaceElement::Vector(vector) => write!(f, "{}", vector),
        }
    }
}

/// The matrix ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct M<T: RingBound> {
    pub(crate) scalar_sub_set: T,
}

impl<T: RingBound> M<T> {
    /// Create a new matrix ring over T
    pub fn new(scalar_sub_set: T) -> Self {
        Self { scalar_sub_set }
    }
}

impl<T: RingBound> Group for M<T> {
    type Element = VectorSpaceElement<TermField<T>, Matrix<TermField<T>>>;

    type ExposantSet = M<T::ExposantSet>;

    fn get_exposant_set(&self) -> Self::ExposantSet {
        self.scalar_sub_set.get_exposant_set().get_matrix_ring()
    }

    fn zero(&self) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.get_term_field().zero())
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (VectorSpaceElement::Scalar(a), VectorSpaceElement::Scalar(b)) => {
                VectorSpaceElement::Scalar(self.scalar_sub_set.get_term_field().add(a, b))
            }
            (VectorSpaceElement::Vector(a), VectorSpaceElement::Vector(b)) => {
                VectorSpaceElement::Vector(a + b)
            }
            _ => panic!("Can't add a scalar and a vector ! ({} + {})", a, b),
        }
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        match a {
            VectorSpaceElement::Scalar(a) => {
                VectorSpaceElement::Scalar(self.scalar_sub_set.get_term_field().neg(a))
            }
            VectorSpaceElement::Vector(a) => VectorSpaceElement::Vector(-a),
        }
    }

    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        match (a, b) {
            (VectorSpaceElement::Scalar(a), VectorSpaceElement::Scalar(b)) => {
                self.scalar_sub_set.get_term_field().partial_cmp(a, b)
            }
            _ => None,
        }
    }
    fn parse_litteral(&self, value: &str) -> Result<Self::Element, String> {
        Ok(VectorSpaceElement::Scalar(
            self.scalar_sub_set.get_term_field().parse_litteral(value)?,
        ))
    }
    fn print(
        &self,
        elem: &Self::Element,
        _: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", elem)
    }
    fn pretty_print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        match elem {
            VectorSpaceElement::Scalar(scalar) => self
                .scalar_sub_set
                .get_term_field()
                .pretty_print(scalar, options),
            VectorSpaceElement::Vector(matrix) => Print::pretty_print(matrix, options),
        }
    }
}

impl<T: RingBound> Ring for M<T> {
    fn one(&self) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.get_term_field().one())
    }

    fn nth(&self, nth: i64) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.get_term_field().nth(nth))
    }

    fn mul(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        todo!()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        match a {
            VectorSpaceElement::Scalar(scalar) => self
                .scalar_sub_set
                .get_term_field()
                .try_inv(scalar)
                .map(VectorSpaceElement::Scalar),
            VectorSpaceElement::Vector(matrix) => matrix.inv().ok().map(VectorSpaceElement::Vector),
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

    fn normalize(&self, mut a: Term<Self>) -> Term<Self> {
        a = match &a {
            Term::Pow(pow)
                if let Term::Value(term::Value {
                    value: VectorSpaceElement::Scalar(ref base),
                    ..
                }) = *pow.base
                    && let Term::Value(term::Value {
                        value: VectorSpaceElement::Scalar(ref exposant),
                        ..
                    }) = **pow.exposant =>
            {
                Term::Value(term::Value::new(
                    VectorSpaceElement::Scalar(Term::Pow(term::Pow::new(
                        base.clone(),
                        exposant.clone(),
                        self.scalar_sub_set,
                    ))),
                    *self,
                ))
            }
            other => other.clone(),
        };
        match &a {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    res.data.iter_mut().for_each(|x| *x = (*x).normalize());
                    return Term::Value(Value::new(VectorSpaceElement::Vector(res), *self));
                }
                VectorSpaceElement::Scalar(scalar) => {
                    return Term::Value(Value::new(
                        VectorSpaceElement::Scalar((*scalar).normalize()),
                        *self,
                    ));
                }
            },
            Term::Pow(term::Pow { base, exposant, .. }) => {
                if let (
                    Term::Value(Value {
                        value: VectorSpaceElement::Vector(matrix),
                        ..
                    }),
                    Term::Value(Value {
                        value: exposant, ..
                    }),
                ) = (&**base, &***exposant)
                    && Group::partial_cmp(
                        &self.get_exposant_set(),
                        exposant,
                        &self.get_exposant_set().zero(),
                    ) == Some(Ordering::Less)
                {
                    return Term::Pow(term::Pow::new(
                        Term::Value(Value::new(
                            VectorSpaceElement::Vector(
                                matrix
                                    .inv()
                                    .unwrap_or_else(|_| panic!("Cannot invert matrix {}", matrix)),
                            ),
                            *self,
                        )),
                        Term::Value(Value::new(
                            self.get_exposant_set().neg(exposant),
                            self.get_exposant_set(),
                        )),
                        *self,
                    ));
                }
            }
            _ => {}
        }
        a
    }

    fn expand(&self, a: Term<Self>) -> Term<Self> {
        match &a {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    res.data.iter_mut().for_each(|x| {
                        *x = (*x).expand();
                    });
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                VectorSpaceElement::Scalar(scalar) => Term::Value(Value::new(
                    VectorSpaceElement::Scalar((*scalar).expand()),
                    *self,
                )),
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
                        *x = (*x).simplify();
                    });
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                VectorSpaceElement::Scalar(scalar) => Term::Value(Value::new(
                    VectorSpaceElement::Scalar((*scalar).simplify()),
                    *self,
                )),
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
pub struct TryCastError(pub(crate) &'static str);
impl Error for TryCastError {}

impl fmt::Display for TryCastError {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format!("Can't cast to this group: {}", self.0))
    }
}

impl<T: RingBound> TryElementFrom<M<T>> for TermField<T> {
    fn try_from_element(value: <M<T> as Group>::Element) -> Result<Self::Element, TryCastError> {
        match value {
            VectorSpaceElement::Scalar(scalar) => Ok(scalar),
            VectorSpaceElement::Vector(_) => Err(TryCastError("Can't cast matrix to scalar")),
        }
    }
}

impl<T: RingBound> TryElementFrom<TermField<T>> for M<T> {
    fn try_from_element(
        value: <TermField<T> as Group>::Element,
    ) -> Result<Self::Element, TryCastError> {
        Ok(VectorSpaceElement::Scalar(value))
    }
}

/// The matrix field, with real coefficients. See [const@R]
pub const M: M<R<Rational>> = M { scalar_sub_set: R };

impl From<usize> for Term<R<Rational>> {
    fn from(value: usize) -> Self {
        Term::Value(Value::new(Rational::from(value), R))
    }
}

impl<T> TryElementFrom<TermField<R<T>>> for R<T>
where
    R<T>: RingBound,
{
    fn try_from_element(
        value: <TermField<R<T>> as Group>::Element,
    ) -> Result<Self::Element, TryCastError> {
        match value {
            Term::Value(value) => Ok(value.get_value()),
            _ => Err(TryCastError("Value is not a constant")),
        }
    }
}
