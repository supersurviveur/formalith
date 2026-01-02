//! The matrix ring `{$M$}`.

use malachite::rational::Rational;
use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display},
};

use crate::{
    field::{Group, PartiallyOrderedSet, Ring, Set, SetParseExpression, real::R},
    matrix::{Matrix, MatrixResult},
    parser::{Parser, ParserError, lexer::TokenKind},
    printer::{PrettyPrint, PrettyPrinter, Print, PrintOptions},
    term::{self, Term, TermSet, Value},
};

/// An enum representing elements inside a vector space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorSpaceElement<T: Set, Vector> {
    /// A scalar element living in `T`
    Scalar(T::Element, T),
    /// A vector living in the vector space
    Vector(Vector),
}

impl<T: Ring> VectorSpaceElement<T, Matrix<T>> {
    /// Compute the determinant of self in set `T`.
    pub fn det(&self) -> MatrixResult<T::Element> {
        match self {
            VectorSpaceElement::Scalar(scalar, _) => Ok(scalar.clone()),
            VectorSpaceElement::Vector(matrix) => matrix.det(),
        }
    }
}

impl<T: PartiallyOrderedSet, Vector: PartialEq> PartialOrd for VectorSpaceElement<T, Vector> {
    #[inline]
    fn partial_cmp(&self, other: &VectorSpaceElement<T, Vector>) -> Option<Ordering> {
        match (self, other) {
            (VectorSpaceElement::Scalar(s1, set), VectorSpaceElement::Scalar(s2, _)) => {
                set.partial_cmp(s1, s2)
            }
            _ => None,
        }
    }
}
impl<T: Set, Vector: Print> Print for VectorSpaceElement<T, Vector> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            VectorSpaceElement::Scalar(scalar, set) => set.print(scalar, options, f),
            VectorSpaceElement::Vector(vector) => vector.print(options, f),
        }
    }
}

impl<T: Set, Vector: PrettyPrint> PrettyPrint for VectorSpaceElement<T, Vector> {
    fn pretty_print(&self, options: &crate::printer::PrintOptions) -> PrettyPrinter {
        match self {
            VectorSpaceElement::Scalar(scalar, set) => set.pretty_print(scalar, options),
            VectorSpaceElement::Vector(vector) => vector.pretty_print(options),
        }
    }
}

impl<T: Set, Vector: PrettyPrint> Display for VectorSpaceElement<T, Vector> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PrettyPrint::fmt(self, &PrintOptions::default(), f)
    }
}

/// The matrix ring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct M<T> {
    pub(crate) scalar_sub_set: T,
}

impl<T> M<T> {
    /// Create a new matrix ring over T
    pub fn new(scalar_sub_set: T) -> Self {
        Self { scalar_sub_set }
    }
}

impl<T: Set> Set for M<T> {
    type Element = VectorSpaceElement<T, Matrix<T>>;

    type ExponantSet = M<T::ExponantSet>;
    type ProductCoefficientSet = M<T::ProductCoefficientSet>;
    fn get_exponant_set(&self) -> Self::ExponantSet {
        self.scalar_sub_set.get_exponant_set().get_matrix_set()
    }
    fn get_coefficient_set(&self) -> Self::ProductCoefficientSet {
        self.scalar_sub_set.get_coefficient_set().get_matrix_set()
    }
    fn print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match elem {
            VectorSpaceElement::Scalar(scalar, _) => self.scalar_sub_set.print(scalar, options, f),
            VectorSpaceElement::Vector(matrix) => Print::print(matrix, options, f),
        }
    }
    fn pretty_print(
        &self,
        elem: &Self::Element,
        options: &crate::printer::PrintOptions,
    ) -> crate::printer::PrettyPrinter {
        match elem {
            VectorSpaceElement::Scalar(scalar, _) => {
                self.scalar_sub_set.pretty_print(scalar, options)
            }
            VectorSpaceElement::Vector(matrix) => PrettyPrint::pretty_print(matrix, options),
        }
    }
    fn element_eq(&self, a: &Self::Element, b: &Self::Element) -> bool {
        a == b
    }
}

impl<T: PartiallyOrderedSet> PartiallyOrderedSet for M<T> {
    fn partial_cmp(&self, a: &Self::Element, b: &Self::Element) -> Option<Ordering> {
        match (a, b) {
            (VectorSpaceElement::Scalar(a, _), VectorSpaceElement::Scalar(b, _)) => {
                self.scalar_sub_set.partial_cmp(a, b)
            }
            _ => None,
        }
    }
}

impl<T: Group> Group for M<T> {
    fn zero(&self) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.zero(), self.scalar_sub_set)
    }

    fn nth(&self, nth: i64) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.nth(nth), self.scalar_sub_set)
    }

    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        match (a, b) {
            (VectorSpaceElement::Scalar(a, _), VectorSpaceElement::Scalar(b, _)) => {
                VectorSpaceElement::Scalar(self.scalar_sub_set.add(a, b), self.scalar_sub_set)
            }
            (VectorSpaceElement::Vector(a), VectorSpaceElement::Vector(b)) => {
                VectorSpaceElement::Vector(a + b)
            }
            _ => panic!("Can't add a scalar and a vector ! ({} + {})", a, b),
        }
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        match a {
            VectorSpaceElement::Scalar(a, _) => {
                VectorSpaceElement::Scalar(self.scalar_sub_set.neg(a), self.scalar_sub_set)
            }
            VectorSpaceElement::Vector(a) => VectorSpaceElement::Vector(-a),
        }
    }
}
impl<T: Ring> Ring for M<T> {
    fn one(&self) -> Self::Element {
        VectorSpaceElement::Scalar(self.scalar_sub_set.one(), self.scalar_sub_set)
    }
    fn mul(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        todo!()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        match a {
            VectorSpaceElement::Scalar(scalar, _) => self
                .scalar_sub_set
                .try_inv(scalar)
                .map(|elem| VectorSpaceElement::Scalar(elem, self.scalar_sub_set)),
            VectorSpaceElement::Vector(matrix) => matrix.inv().ok().map(VectorSpaceElement::Vector),
        }
    }

    fn as_fraction(&self, a: &Self::Element) -> (Self::Element, Self::Element) {
        let num = match &a {
            VectorSpaceElement::Vector(matrix) => {
                let mut res = matrix.clone();
                res.data.iter_mut().for_each(|x| {
                    *x = self
                            .scalar_sub_set
                            .normalize(Term::Value(Value::new(x.clone(), self.scalar_sub_set)))
                            .as_value()
                            .expect("Unified expression is not a value and should be converted inside the matrix ring")
                            .value
                });
                VectorSpaceElement::Vector(res)
            }
            _ => a.clone(),
        };
        (num, self.one())
    }

    fn normalize(&self, mut a: Term<Self>) -> Term<Self> {
        a = if let Term::Pow(pow) = &a
            && let Term::Value(term::Value {
                value: VectorSpaceElement::Scalar(ref base, _),
                ..
            }) = *pow.base
            && let Term::Value(term::Value {
                value: VectorSpaceElement::Scalar(ref exponant, _),
                ..
            }) = **pow.exponant
        {
            Term::Value(Value::new(VectorSpaceElement::Scalar(self.scalar_sub_set.normalize(Term::Pow(term::Pow::new(
                Term::Value(Value::new(base.clone(), self.scalar_sub_set)),
                Term::Value(Value::new(
                    exponant.clone(),
                    self.scalar_sub_set.get_exponant_set(),
                )),
                self.scalar_sub_set,
            ))).as_value().expect("Normalized expression is not a value and should be converted inside the matrix ring").value,
                self.scalar_sub_set),*self))
        } else {
            a
        };
        match &a {
            Term::Value(value) => match &value.value {
                VectorSpaceElement::Vector(matrix) => {
                    let mut res = matrix.clone();
                    res.data.iter_mut().for_each(|x| *x = self
                            .scalar_sub_set
                            .normalize(Term::Value(Value::new(x.clone(), self.scalar_sub_set)))
                            .as_value()
                            .expect("Normalized expression is not a value and should be converted inside the matrix ring")
                            .value);
                    return Term::Value(Value::new(VectorSpaceElement::Vector(res), *self));
                }
                VectorSpaceElement::Scalar(scalar, _) => {
                    let expanded = self
                        .scalar_sub_set
                        .normalize(Term::Value(Value::new(scalar.clone(), self.scalar_sub_set)));
                    match expanded {
                        Term::Value(Value { value, .. }) => {
                            return Term::Value(Value::new(
                                VectorSpaceElement::Scalar(value, self.scalar_sub_set),
                                *self,
                            ));
                        }
                        _ => unimplemented!(
                            "Normalized expression is not a value and should be converted inside the matrix ring"
                        ),
                    }
                }
            },
            Term::Pow(term::Pow { base, exponant, .. }) => {
                if let (
                    Term::Value(Value {
                        value: VectorSpaceElement::Vector(matrix),
                        ..
                    }),
                    Term::Value(Value {
                        value: exponant, ..
                    }),
                ) = (&**base, &***exponant)
                    && self.get_exponant_set().is_strictly_negative(exponant)
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
                            self.get_exponant_set().neg(exponant),
                            self.get_exponant_set(),
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
                        *x = self
                            .scalar_sub_set
                            .expand(Term::Value(Value::new(x.clone(), self.scalar_sub_set)))
                            .as_value()
                            .expect("Expanded expression is not a value and should be converted inside the matrix ring")
                            .value;
                    });
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                VectorSpaceElement::Scalar(scalar, _) => {
                    let expanded = self
                        .scalar_sub_set
                        .expand(Term::Value(Value::new(scalar.clone(), self.scalar_sub_set)));
                    match expanded {
                        Term::Value(Value { value, .. }) => Term::Value(Value::new(
                            VectorSpaceElement::Scalar(value, self.scalar_sub_set),
                            *self,
                        )),
                        _ => unimplemented!(
                            "Expanded expression is not a value and should be converted inside the matrix ring"
                        ),
                    }
                }
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
                        *x = self
                            .scalar_sub_set
                            .simplify(Term::Value(Value::new(x.clone(), self.scalar_sub_set)))
                            .as_value()
                            .expect("Simplified expression is not a value and should be converted inside the matrix ring")
                            .value;
                    });
                    Term::Value(Value::new(VectorSpaceElement::Vector(res), *self))
                }
                VectorSpaceElement::Scalar(scalar, _) => {
                    let expanded = self
                        .scalar_sub_set
                        .simplify(Term::Value(Value::new(scalar.clone(), self.scalar_sub_set)));
                    match expanded {
                        Term::Value(Value { value, .. }) => Term::Value(Value::new(
                            VectorSpaceElement::Scalar(value, self.scalar_sub_set),
                            *self,
                        )),
                        _ => unimplemented!(
                            "Simplified expression is not a value and should be converted inside the matrix ring"
                        ),
                    }
                }
            },
            _ => a,
        }
    }
}

impl<T: SetParseExpression<N>, N> SetParseExpression<N> for M<T> {
    fn parse_literal(&self, parser: &mut Parser) -> Result<Option<Self::Element>, String> {
        Ok(self
            .scalar_sub_set
            .parse_literal(parser)?
            .map(|parsed| VectorSpaceElement::Scalar(parsed, self.scalar_sub_set)))
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
                        if let Some(element) = parser.parse_literal(self.scalar_sub_set)? {
                            lines.push(element);
                        } else {
                            return Ok(None);
                        }
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
                        self.scalar_sub_set,
                    )),
                    *self,
                ))))
            }
            _ => Ok(None),
        }
    }
}

/// The matrix field, with real coefficients. See [const@R]
pub const M: M<TermSet<R<Rational>>> = M {
    scalar_sub_set: TermSet::new(R),
};
