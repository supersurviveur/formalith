//! Complexity is `{$f(x) = (1+ epsilon)abs(x)$}`, where `{$abs(epsilon) < 2^-54$}`, `{$sum x^2$}`
//!
//! $ f(x) $

#![warn(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

pub mod combinatorics;
pub mod context;
pub mod field;
pub mod matrix;
pub mod parser;
pub mod polynom;
pub mod printer;
pub mod term;
