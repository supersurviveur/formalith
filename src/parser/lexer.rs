use std::{fmt::Display, str::Chars};

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub len: usize,
}

impl Token {
    pub fn new(kind: TokenKind, len: usize) -> Token {
        Token { kind, len }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum TokenKind {
    Whitespace,

    OpenParen,
    CloseParen,
    OpenBracket,
    CloseBracket,

    Ident(String),

    // Eq,
    // EqEq,
    // Bang,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,

    Literral { kind: LiterralTokenKind },

    Comma,
    Eof,
    Unknown,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum LiterralTokenKind {
    Int,
    Float,
    Bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Lexer<'a> {
    len_remaining: usize,
    pos: usize,
    chars: Chars<'a>,
    input: &'a str,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Lexer {
            len_remaining: input.len(),
            chars: input.chars(),
            pos: 0,
            input,
        }
    }
}
impl Lexer<'_> {
    /// Returns amount of already consumed symbols.
    fn pos_within_token(&self) -> usize {
        self.len_remaining - self.chars.as_str().len()
    }

    /// Resets the number of bytes consumed to 0.
    fn reset_pos_within_token(&mut self) {
        self.pos = self.current_pos();
        self.len_remaining = self.chars.as_str().len();
    }

    // Return the current position in the input string.
    fn current_pos(&self) -> usize {
        self.pos + self.pos_within_token()
    }

    fn peek_char(&self) -> char {
        self.chars.clone().next().unwrap_or('\0')
    }
    fn eof(&mut self) -> bool {
        self.chars.clone().next().is_none()
    }
    fn lex(&mut self) -> Token {
        // Stop the lexer at the end of the file
        if self.eof() {
            return Token::new(TokenKind::Eof, 0);
        }
        let token_kind = match self.chars.next().unwrap() {
            '/' => TokenKind::Slash,

            c if c.is_ascii_whitespace() => self.consume_whitespaces(),
            '0'..='9' => self.consume_number(),

            '+' => TokenKind::Plus,
            '-' => TokenKind::Minus,

            '*' => TokenKind::Star,
            ',' => TokenKind::Comma,
            '^' => TokenKind::Caret,
            // '!' => TokenKind::Bang,
            // '=' => match self.peek_char() {
            //     '=' => {
            //         self.chars.next();
            //         TokenKind::EqEq
            //     }
            //     _ => TokenKind::Eq,
            // },
            '{' => TokenKind::OpenBracket,
            '}' => TokenKind::CloseBracket,
            '(' => TokenKind::OpenParen,
            ')' => TokenKind::CloseParen,
            '[' => TokenKind::OpenBracket,
            ']' => TokenKind::CloseBracket,

            c if c.is_ascii_alphabetic() || c == '_' => self.consume_ident(),
            _ => TokenKind::Unknown,
        };
        let len = self.pos_within_token();
        self.reset_pos_within_token();
        Token::new(token_kind, len)
    }
    fn consume_whitespaces(&mut self) -> TokenKind {
        while self.peek_char().is_ascii_whitespace() && !self.eof() {
            self.chars.next();
        }
        TokenKind::Whitespace
    }
    fn consume_number(&mut self) -> TokenKind {
        while self.peek_char().is_digit(10) && !self.eof() {
            self.chars.next();
        }
        if self.peek_char() == '.' {
            self.chars.next();
            while self.peek_char().is_digit(10) && !self.eof() {
                self.chars.next();
            }
            TokenKind::Literral {
                kind: LiterralTokenKind::Float,
            }
        } else {
            TokenKind::Literral {
                kind: LiterralTokenKind::Int,
            }
        }
    }
    fn consume_ident(&mut self) -> TokenKind {
        while self.peek_char().is_ascii_alphanumeric() || self.peek_char() == '_' && !self.eof() {
            self.chars.next();
        }
        let ident = &self.input[self.pos..self.current_pos()];
        match ident {
            "true" | "false" => TokenKind::Literral {
                kind: LiterralTokenKind::Bool,
            },
            _ => TokenKind::Ident(ident.to_string()),
        }
    }
}

impl Iterator for Lexer<'_> {
    type Item = Token;
    fn next(&mut self) -> Option<Token> {
        let token = self.lex();
        match token.kind {
            TokenKind::Eof => None,
            _ => Some(token),
        }
    }
}

impl Display for Lexer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let new = self.clone();
        for token in new {
            write!(f, "{:?}\n", token)?;
        }
        Ok(())
    }
}
