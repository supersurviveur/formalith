//! The real field `{$RR$}`.

use std::{cmp::Ordering, marker::PhantomData};

use malachite::{
    Integer, Natural, Rational,
    base::num::{
        arithmetic::traits::{CheckedRoot, Parity, Pow, Sign},
        basic::traits::{NegativeOne, One, Zero},
        conversion::traits::FromSciString,
    },
};

use crate::{
    combinatorics,
    context::{Context, Symbol},
    field::{
        Derivable, Field, Group, PartiallyOrderedSet, Ring, Set, SetParseExpression, matrix::M,
    },
    parser::Parser,
    printer::PrettyPrinter,
    term::{self, Fun, Mul, Normalize, Term, TermSet, Value},
};

/// The real field `{$RR$}`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct R<T> {
    phantom: PhantomData<T>,
}

impl<T: Clone> Copy for R<T> {}

impl Set for R<Rational> {
    type Element = Rational;
    type ExponantSet = R<Rational>;
    type ProductCoefficientSet = R<Rational>;
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
    fn element_eq(&self, a: &Self::Element, b: &Self::Element) -> bool {
        a == b
    }
}

impl PartiallyOrderedSet for R<Rational> {
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        a.partial_cmp(b)
    }
}

impl Group for R<Rational> {
    fn zero(&self) -> Self::Element {
        Rational::ZERO
    }

    fn nth(&self, nth: i64) -> <Self::ProductCoefficientSet as Set>::Element {
        Rational::const_from_signed(nth)
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a
    }
}

impl Ring for R<Rational> {
    fn one(&self) -> Self::Element {
        Rational::ONE
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
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
            Term::Pow(term::Pow { base, exponant, .. }) => match (&**base, &***exponant) {
                (Term::Pow(base), Term::Value(exponant)) => {
                    let (numerator, denominator) = exponant.value.to_numerator_and_denominator();
                    let mut res = base.clone();

                    // Numerator and sign can always be passed to the product
                    // If denominator is odd, we can pass it to the product too
                    let (passed_denominator, denominator) = if denominator.odd() {
                        (denominator, Natural::ONE)
                    } else {
                        (Natural::ONE, denominator)
                    };
                    let mul = term::Mul::new(
                        Rational::from_sign_and_naturals(
                            exponant.value.sign() == Ordering::Greater,
                            numerator.clone(),
                            passed_denominator,
                        ),
                        vec![(**base.exponant).clone()],
                        self.get_exponant_set(),
                    );

                    let mul = Term::Mul(mul).normalize();
                    let mut res = if mul.is_one() {
                        *res.base.clone()
                    } else if mul.is_zero() {
                        Term::Value(Value::new(Rational::ONE, *self))
                    } else {
                        *res.exponant = mul.into();
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
                (Term::Value(base), Term::Value(exponant)) => {
                    let exponant = &exponant.value;
                    let mut res = base.value.clone();
                    if exponant.sign() == Ordering::Less {
                        res = Pow::pow(res, -1_i64);
                    }
                    let (mut num, den) = exponant.to_numerator_and_denominator();
                    if let Ok(exponant) = i64::try_from(&num) {
                        res = Pow::pow(res, exponant);
                        num = Natural::ONE;
                        if let Ok(root_exponant) = i64::try_from(&den)
                            && let Some(root_free) = (&res).checked_root(root_exponant)
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
                    if let Some(Term::Value(matrix)) = fun.get_arg::<M<TermSet<Self>>>()
                        && let Ok(res) = matrix.value.det()
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
            Term::Pow(term::Pow { base, exponant, .. }) => {
                if let (Term::Add(add), Term::Value(value)) = (&**base, &***exponant) {
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
                } else if let (Term::Mul(mul), _) = (&**base, &**exponant) {
                    return Term::Mul(Mul::new(
                        self.get_coefficient_set().nth(1),
                        mul.into_iter()
                            .map(|x| term::Pow::new(x.clone(), (**exponant).clone(), *self).into())
                            .chain([term::Pow::new(
                                Term::Value(Value::new(mul.coefficient.clone(), *self)),
                                (**exponant).clone(),
                                *self,
                            )
                            .into()])
                            .collect(),
                        *self,
                    ))
                    .normalize();
                }
                a
            }
            _ => a,
        }
    }
}

impl<N> SetParseExpression<N> for R<Rational> {
    fn parse_literal(&self, parser: &mut Parser) -> Result<Option<Self::Element>, String> {
        parser.is_literal_and(|value| {
            Ok(Some(Rational::from_sci_string(value).ok_or(format!(
                "Failed to parse \"{value}\" to malachite::Integer",
            ))?))
        })
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
