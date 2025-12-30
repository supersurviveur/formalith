//! Parser implementation, used in [crate::parse] and [crate::try_parse] macros.

pub mod lexer;

use std::error::Error;
use std::fmt::{Debug, Display};
use std::iter::Peekable;
use std::ops::{Range, Sub};

use crate::context::{Context, Symbol};
use crate::field::{Group, M, Ring, RingBound, Set, SetParseExpression, SetParseExpressionBound};
use crate::term::{Fun, Mul, Normalize, SymbolTerm, Term, Value};

use lexer::{Lexer, Token, TokenKind};
use typenum::{Diff, U0, U1, U10, UInt};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Add,
    Substract,
    Multiply,
    Divide,
    Pow,
}
impl Op {
    pub fn get_priority(&self) -> usize {
        match self {
            Op::Pow => 14,
            Op::Multiply | Op::Divide => 13,
            Op::Add | Op::Substract => 12,
        }
    }
}

/// Represent an error raised by the parser
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ParserError {
    message: String,
}

impl ParserError {
    /// Create a new parser error.
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

impl Error for ParserError {}

impl Display for ParserError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// A parser over a string input.
#[derive(Debug, Clone)]
pub struct Parser<'a> {
    input: &'a str,
    lexer: Peekable<Lexer<'a>>,
    prev_token: Token,
    /// The current token
    pub token: Token,
    start_span: usize,
    end_span: usize,
}

impl<'a> Parser<'a> {
    /// Create a new parser over the `input` string.
    pub fn new(input: &'a str) -> Self {
        let mut parser = Parser {
            input,
            lexer: Lexer::new(input).peekable(),
            prev_token: Token::new(TokenKind::Unknown, 0),
            token: Token::new(TokenKind::Unknown, 0),
            start_span: 0,
            end_span: 0,
        };
        parser.next_token();
        parser
    }
}

/// Trait used by parser to implement some methods with specialization.
pub trait ParserTraitBounded<E: Set, N> {
    /// Parse an expression, which can be a literal, a sum, a group ()...
    fn parse_expression_bounded(
        &mut self,
        priority: usize,
        set: E,
    ) -> Result<Option<Term<E>>, ParserError>;
}

impl Parser<'_> {
    /// Return the next lexer token and advance current span
    fn next_token_internal(&mut self) -> Option<Token> {
        let next = self.lexer.next();
        self.start_span = self.end_span;
        if let Some(span) = &next {
            self.end_span += span.len;
        }
        next
    }
    /// Update the current and previous token, skipping unnecessary tokens
    pub fn next_token(&mut self) {
        let mut next = self.next_token_internal();
        loop {
            match next {
                None => {
                    next = Some(Token::new(TokenKind::Eof, 0));
                    break;
                }
                Some(ref token) => match token.kind {
                    TokenKind::Whitespace => {}
                    _ => break,
                },
            }
            next = self.next_token_internal()
        }
        self.prev_token = std::mem::replace(&mut self.token, next.unwrap())
    }
    fn span(&mut self) -> Range<usize> {
        self.start_span..self.end_span
    }
    /// Eat a token with a specific kind, raising an error if the current token kind doesn't match.  
    pub fn expect_token(&mut self, kind: TokenKind) -> Result<(), ParserError> {
        match &self.token.kind {
            k if *k == kind => {
                self.next_token();
                Ok(())
            }
            _ => Err(ParserError::new(format!(
                "expect {:?}, found {:?}",
                kind, self.token.kind
            ))),
        }
    }
    /// Return the next operator if it exists and consume it
    fn get_op(&mut self) -> Option<Op> {
        match self.token.kind {
            TokenKind::Plus => Some(Op::Add),
            TokenKind::Minus => Some(Op::Substract),
            TokenKind::Star => Some(Op::Multiply),
            TokenKind::Slash => Some(Op::Divide),
            TokenKind::Caret => Some(Op::Pow),
            _ => None,
        }
    }
    /// Parse the parser's string, returning the parsed mathematical expression.
    pub fn parse<E: RingBound + SetParseExpressionBound>(
        &mut self,
        set: E,
    ) -> Result<Term<E>, ParserError> {
        Ok(
            ParserTraitBounded::<E, U10>::parse_expression_bounded(self, 0, set)?
                .ok_or(ParserError::new("Input was empty".into()))?
                .normalize(),
        )
    }

    fn parse_symbol<E: Set>(&mut self, set: E) -> Result<Option<Term<E>>, ParserError> {
        match &self.token.kind {
            TokenKind::Ident(sym) => {
                let tmp = sym.clone();
                self.next_token();
                Ok(Some(Term::Symbol(SymbolTerm::new(Symbol::new(tmp), set))))
            }
            _ => Ok(None),
        }
    }
    fn parse_group<E: Set, N>(&mut self, set: E) -> Result<Option<Term<E>>, ParserError>
    where
        Self: ParserTraitBounded<E, N>,
    {
        match &self.token.kind {
            TokenKind::OpenParen => {
                self.next_token();
                let content =
                    ParserTraitBounded::<E, N>::parse_expression_bounded(self, 0, set)?.unwrap();
                self.expect_token(TokenKind::CloseParen)?;
                Ok(Some(content))
            }
            _ => Ok(None),
        }
    }

    fn parse_args<U: Set, N>(&mut self, arg_set: U) -> Result<Vec<Term<U>>, ParserError>
    where
        Self: ParserTraitBounded<U, N>,
    {
        let mut args = vec![];
        args.push(self.parse_expression_bounded(0, arg_set)?.unwrap());
        while self.token.kind == TokenKind::Comma {
            self.next_token();
            args.push(self.parse_expression_bounded(0, arg_set)?.unwrap())
        }
        Ok(args)
    }
}

impl<E: Set> ParserTraitBounded<E, U0> for Parser<'_> {
    fn parse_expression_bounded(&mut self, _: usize, _: E) -> Result<Option<Term<E>>, ParserError> {
        Err(ParserError::new(
            "Cannot recurse inside parser more than 10 times".to_string(),
        ))
    }
}

impl<E: Set + SetParseExpression<UInt<U, B>>, U, B> ParserTraitBounded<E, UInt<U, B>> for Parser<'_>
where
    UInt<U, B>: Sub<U1>,
    Self: ParserTraitBounded<M<E>, U>,
    Self: ParserTraitBounded<<E as crate::field::Set>::ExponantSet, Diff<UInt<U, B>, U1>>,
    E::ProductCoefficientSet: SetParseExpression<UInt<U, B>>,
{
    fn parse_expression_bounded(
        &mut self,
        priority: usize,
        set: E,
    ) -> Result<Option<Term<E>>, ParserError> {
        let mut self_in_coefficient_set = self.clone();
        let node = set.parse_expression(self)?;
        let node = node.map_or_else(
            || ParseUnary::<E, UInt<U, B>>::parse_unary(self, priority, set),
            |n| Ok(Some(n)),
        )?;
        let node_in_coefficient_set =
            self_in_coefficient_set.parse_literal(set.get_coefficient_set());
        let node = node.map_or_else(|| self.parse_literal_term(set), |n| Ok(Some(n)))?;
        let node = node.map_or_else(|| self.parse_symbol(set), |n| Ok(Some(n)))?;
        let node = node.map_or_else(|| self.parse_group::<E, UInt<U, B>>(set), |n| Ok(Some(n)))?;

        match (node, node_in_coefficient_set) {
            (None, Err(_)) | (None, Ok(None)) => Ok(None),
            (Some(mut node), node_in_coefficient_set) => {
                if let (TokenKind::OpenParen, Term::Symbol(symbol)) = (&self.token.kind, &node) {
                    self.next_token();
                    let arg_set = match symbol.symbol {
                        Context::DET => Term::Fun(Box::new(Fun::new(
                            symbol.symbol,
                            self.parse_args::<M<E>, U>(set.get_matrix_set())?,
                            set,
                        ))),
                        _ => Term::Fun(Box::new(Fun::new(
                            symbol.symbol,
                            self.parse_args::<E, UInt<U, B>>(set)?,
                            set,
                        ))),
                    };
                    node = arg_set
                }
                let mut op = self.get_op();
                while self.token.kind != TokenKind::Eof
                    && op.is_some()
                    && priority < op.as_ref().unwrap().get_priority()
                {
                    self.next_token();
                    let (new_node, parsed) = ParseBinary::<E, UInt<U, B>>::parse_binary_expr(
                        self,
                        node,
                        node_in_coefficient_set.clone(),
                        op.unwrap(),
                        set,
                    )?;
                    node = new_node;
                    if !parsed {
                        break;
                    }
                    op = self.get_op();
                }
                Ok(Some(node))
            }
            _ => todo!(),
        }
    }
}
trait ParseBinary<E: Set, N> {
    fn parse_binary_expr(
        &mut self,
        current: Term<E>,
        current_in_coefficient_set: Result<
            Option<<<E as Set>::ProductCoefficientSet as Set>::Element>,
            ParserError,
        >,
        op: Op,
        set: E,
    ) -> Result<(Term<E>, bool), ParserError>;
}

impl<E: Set, N> ParseBinary<E, N> for Parser<'_> {
    default fn parse_binary_expr(
        &mut self,
        current: Term<E>,
        _current_in_coefficient_set: Result<
            Option<<<E as Set>::ProductCoefficientSet as Set>::Element>,
            ParserError,
        >,
        _op: Op,
        _set: E,
    ) -> Result<(Term<E>, bool), ParserError> {
        Ok((current, false))
    }
}

impl<E: Ring, N> ParseBinary<E, N> for Parser<'_>
where
    N: Sub<U1>,
    Self: ParserTraitBounded<E, N>,
    Self: ParserTraitBounded<E::ExponantSet, Diff<N, U1>>,
{
    fn parse_binary_expr(
        &mut self,
        current: Term<E>,
        current_in_coefficient_set: Result<
            Option<<<E as Set>::ProductCoefficientSet as Set>::Element>,
            ParserError,
        >,
        op: Op,
        set: E,
    ) -> Result<(Term<E>, bool), ParserError> {
        if Op::Pow == op {
            if let Some(right) =
                ParserTraitBounded::<E::ExponantSet, Diff<N, U1>>::parse_expression_bounded(
                    self,
                    op.get_priority(),
                    set.get_exposant_set(),
                )?
            {
                return Ok((current.pow(&right), true));
            }
        } else if let Some(right) =
            ParserTraitBounded::<E, N>::parse_expression_bounded(self, op.get_priority(), set)?
        {
            return Ok((
                match op {
                    Op::Add => current + right,
                    Op::Substract => current - right,
                    Op::Multiply => {
                        if let Ok(Some(current)) = current_in_coefficient_set {
                            Term::Mul(Mul::new(current, vec![right], set)).normalize()
                        } else {
                            current * right
                        }
                    }
                    Op::Divide => current / right,
                    Op::Pow => unreachable!(),
                },
                true,
            ));
        }
        Ok((current, false))
    }
}
impl Parser<'_> {
    fn parse_literal_term<E: Set>(&mut self, set: E) -> Result<Option<Term<E>>, ParserError> {
        Ok(self
            .parse_literal(set)?
            .map(|lit| Term::Value(Value::new(lit, set))))
    }
    fn parse_literal<E: Set>(&mut self, set: E) -> Result<Option<E::Element>, ParserError> {
        let content = &self.input[self.span()];
        match &self.token.kind {
            TokenKind::Literal { .. } => {
                let lit = set.parse_literal(content).map_err(ParserError::new)?;
                self.next_token();
                Ok(Some(lit))
            }
            _ => Ok(None),
        }
    }
}

trait ParseUnary<E: Set, N> {
    fn parse_unary(&mut self, _priority: usize, _set: E) -> Result<Option<Term<E>>, ParserError>;
}

impl<E: Set, N> ParseUnary<E, N> for Parser<'_> {
    default fn parse_unary(
        &mut self,
        _priority: usize,
        _set: E,
    ) -> Result<Option<Term<E>>, ParserError> {
        Ok(None)
    }
}

impl<E: Group, N> ParseUnary<E, N> for Parser<'_>
where
    Self: ParserTraitBounded<E, N>,
{
    fn parse_unary(&mut self, priority: usize, set: E) -> Result<Option<Term<E>>, ParserError> {
        match self.get_op() {
            Some(Op::Substract) => {
                self.next_token();
                Ok(
                    ParserTraitBounded::<E, N>::parse_expression_bounded(self, priority, set)?
                        .map(|term| -term),
                )
            }
            _ => Ok(None),
        }
    }
}

/// Try parsing the given input inside the given set, returning a `Result`. See [crate::parse].
#[macro_export(local_inner_macros)]
macro_rules! try_parse {
    ( $x:expr, $f:expr ) => {
        $crate::parser::Parser::new($x.as_ref()).parse($f)
    };
}

/// Parse the given input inside the given set, panic if it is incorrect.
#[macro_export(local_inner_macros)]
macro_rules! parse {
    ( $($args:tt)* ) => {
        try_parse!($($args)*).unwrap()
    };
}
