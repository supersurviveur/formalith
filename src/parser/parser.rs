//! Parser implementation.

use std::error::Error;
use std::fmt::{Debug, Display};
use std::iter::Peekable;
use std::ops::Range;

use crate::context::{Context, Symbol};
use crate::field::{M, RingBound};
use crate::term::{Fun, SymbolTerm, Term, Value};

use super::lexer::{Lexer, Token, TokenKind};

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
#[derive(Debug)]
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
pub trait ParserTrait<E: RingBound> {
    /// Parse an expression, which can be a literal, a sum, a group ()...
    fn parse_expression(&mut self, priority: usize, set: E)
    -> Result<Option<Term<E>>, ParserError>;
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
    pub fn parse<E: RingBound>(&mut self, set: E) -> Result<Term<E>, ParserError> {
        Ok(self
            .parse_expression(0, set)?
            .ok_or(ParserError::new("Input was empty".into()))?
            .normalize())
    }

    fn parse_symbol<E: RingBound>(&mut self, set: E) -> Result<Option<Term<E>>, ParserError> {
        match &self.token.kind {
            TokenKind::Ident(sym) => {
                let tmp = sym.clone();
                self.next_token();
                Ok(Some(Term::Symbol(SymbolTerm::new(Symbol::new(tmp), set))))
            }
            _ => Ok(None),
        }
    }
    fn parse_group<E: RingBound>(&mut self, set: E) -> Result<Option<Term<E>>, ParserError> {
        match &self.token.kind {
            TokenKind::OpenParen => {
                self.next_token();
                let content = self.parse_expression(0, set)?.unwrap();
                self.expect_token(TokenKind::CloseParen)?;
                Ok(Some(content))
            }
            _ => Ok(None),
        }
    }

    fn parse_args<U: RingBound>(&mut self, arg_set: U) -> Result<Vec<Term<U>>, ParserError> {
        let mut args = vec![];
        args.push(self.parse_expression(0, arg_set)?.unwrap());
        while self.token.kind == TokenKind::Comma {
            self.next_token();
            args.push(self.parse_expression(0, arg_set)?.unwrap())
        }
        Ok(args)
    }
}

impl<E: RingBound> ParserTrait<M<M<M<E>>>> for Parser<'_> {
    fn parse_expression(
        &mut self,
        _: usize,
        _: M<M<M<E>>>,
    ) -> Result<Option<Term<M<M<M<E>>>>>, ParserError> {
        Err(ParserError::new(
            "Cannot recurse inside matrix ring more than 2 times".to_string(),
        ))
    }
}
impl<E: RingBound> ParserTrait<E> for Parser<'_> {
    default fn parse_expression(
        &mut self,
        priority: usize,
        set: E,
    ) -> Result<Option<Term<E>>, ParserError> {
        let node = set.parse_expression(self)?;
        let node = node.map_or_else(|| self.parse_unary(priority, set), |n| Ok(Some(n)))?;
        let node = node.map_or_else(|| self.parse_literal(set), |n| Ok(Some(n)))?;
        let node = node.map_or_else(|| self.parse_symbol(set), |n| Ok(Some(n)))?;
        let node = node.map_or_else(|| self.parse_group(set), |n| Ok(Some(n)))?;
        match node {
            None => Ok(None),
            Some(mut node) => {
                if let (TokenKind::OpenParen, Term::Symbol(symbol)) = (&self.token.kind, &node) {
                    self.next_token();
                    let arg_set = match symbol.symbol {
                        Context::DET => Term::Fun(Box::new(Fun::new(
                            symbol.symbol,
                            self.parse_args(set.get_matrix_ring())?,
                            set,
                        ))),
                        _ => Term::Fun(Box::new(Fun::new(
                            symbol.symbol,
                            self.parse_args(set)?,
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
                    let (new_node, parsed) = self.parse_binary_expr(node, op.unwrap(), set)?;
                    node = new_node;
                    if !parsed {
                        break;
                    }
                    op = self.get_op();
                }
                Ok(Some(node))
            }
        }
    }
}
impl Parser<'_> {
    fn parse_binary_expr<E: RingBound>(
        &mut self,
        current: Term<E>,
        op: Op,
        set: E,
    ) -> Result<(Term<E>, bool), ParserError> {
        if let Op::Pow = op {
            if let Some(right) = self.parse_expression(op.get_priority(), set.get_exposant_set())? {
                return Ok((current.pow(&right), true));
            }
        } else if let Some(right) = self.parse_expression(op.get_priority(), set)? {
            return Ok((
                match op {
                    Op::Add => current + right,
                    Op::Substract => current - right,
                    Op::Multiply => current * right,
                    Op::Divide => current / right,
                    Op::Pow => unreachable!(),
                },
                true,
            ));
        }
        Ok((current, false))
    }

    fn parse_literal<E: RingBound>(&mut self, set: E) -> Result<Option<Term<E>>, ParserError> {
        let content = &self.input[self.span()];
        match &self.token.kind {
            TokenKind::Literal { .. } => {
                let lit = Term::Value(Value::new(
                    set.parse_litteral(content)
                        .map_err(ParserError::new)?,
                    set,
                ));
                self.next_token();
                Ok(Some(lit))
            }
            _ => Ok(None),
        }
    }
    fn parse_unary<E: RingBound>(
        &mut self,
        priority: usize,
        set: E,
    ) -> Result<Option<Term<E>>, ParserError> {
        match self.get_op() {
            Some(Op::Substract) => {
                self.next_token();
                Ok(self.parse_expression(priority, set)?.map(|term| -term))
            }
            _ => Ok(None),
        }
    }
}

/// Try parsing the given input inside the given set, returning a `Result`. See [crate::parse].
#[macro_export(local_inner_macros)]
macro_rules! try_parse {
    ( $x:expr, $f:expr ) => {
        $crate::parser::parser::Parser::new($x.as_ref()).parse($f)
    };
}

/// Parse the given input inside the given set, panic if it is incorrect.
#[macro_export(local_inner_macros)]
macro_rules! parse {
    ( $($args:tt)* ) => {
        try_parse!($($args)*).unwrap()
    };
}
