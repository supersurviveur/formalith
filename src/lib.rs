//! Complexity is $f(x) = (1+ epsilon)abs(x)$, where $abs(epsilon) < 2^-54$, $ x^2 $
//!
//! $ f(x) $

#![warn(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]
// Specialization is used at many places, but is especially usefull for :
// - Implementing `TryElementCast` between every types
// - Breaking infinite types cycle by implementing function (which are just call to panic in most cases) after some recursion (e.g. `M<M<M<T>>>`, see [Parser::parse_expression])
#![allow(incomplete_features)]
#![feature(specialization)]

pub mod combinatorics;
pub mod context;
pub mod field;
pub mod matrix;
pub mod parser;
pub mod polynom;
pub mod printer;
pub mod term;
