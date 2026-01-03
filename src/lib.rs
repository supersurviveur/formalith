//! Complexity is `{$f(x) = (1+ epsilon)abs(x)$}`, where `{$abs(epsilon) < 2^-54$}`, `{$sum x^2$}`
//!
//! $ f(x) $

// Enable some additional lints for code and documentation clarity
#![warn(
    rustdoc::all,
    missing_docs,
    clippy::cargo,
    clippy::pedantic,
    // clippy::nursery
)]
// Allow this lint since it cause a stack overflow in clippy, probably due to infinite types.
// A bug report should probably be filled
#![allow(clippy::significant_drop_in_scrutinee)]
// Specialization is incomplete
#![allow(incomplete_features)]
// Usefull features. Since specialization is needed and therefore nightly edition too, we use other usefull features.
#![feature(specialization, iter_order_by)]

pub mod combinatorics;
pub mod context;
pub mod field;
pub mod matrix;
pub mod parser;
pub mod polynom;
pub mod printer;
pub mod term;
pub mod traits;
