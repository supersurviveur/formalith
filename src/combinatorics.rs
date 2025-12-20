//! Combinatorics iterators and functions

use std::marker::PhantomData;

use malachite::{
    base::num::basic::traits::{One, Zero},
    integer::exhaustive::integer_increasing_inclusive_range,
    Integer,
};

/// Iterator over compositions.
///
/// ```no_test
/// use formalith::combinatorics::CompositionIterator;
///
/// let mut iterator = CompositionIterator::new(3, 2);
/// while let Some(composition) = iterator.next() {
///     println!("{:?}", composition);
/// }
/// ```
/// will print
pub struct CompositionIterator<'a> {
    indices: Vec<usize>,
    k: usize,
    init: bool,
    _phantom: PhantomData<&'a usize>,
}

impl CompositionIterator<'_> {
    /// Create a new composition iterator.
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            indices: vec![0; n],
            k,
            init: false,
            _phantom: PhantomData {},
        }
    }

    /// Yield the next composition.
    pub fn next_composition(&mut self) -> Option<&[usize]> {
        if self.indices.is_empty() {
            return None;
        }

        if !self.init {
            self.init = true;
            self.indices[0] = self.k;
            return Some(&self.indices);
        }

        if self.k == 0 {
            return None;
        }

        let mut i = self.indices.len() - 1;
        while self.indices[i] == 0 {
            i -= 1;
        }

        let mut last_value = 0;
        if i == self.indices.len() - 1 {
            // End of the vector was reached
            last_value = self.indices[i];
            self.indices[i] = 0;

            if self.indices.len() == 1 {
                return None;
            }

            i -= 1;
            while self.indices[i] == 0 {
                if i == 0 {
                    return None;
                }

                i -= 1;
            }
        }

        self.indices[i] -= 1;
        self.indices[i + 1] = last_value + 1;

        Some(&self.indices)
    }
}

/// General trait to do combinatorics on any integer type.
pub trait Combinatorics: Sized {
    /// Compute a Newton binomial
    fn binom(n: Self, k: Self) -> Self;
    /// Compute a Newton multinomial
    fn multinom(k: &[Self]) -> Self;
}

impl Combinatorics for Integer {
    fn binom(n: Self, mut k: Self) -> Self {
        if n < 0 || k < 0 || k > n {
            return Integer::ZERO;
        }

        if k > &n / Integer::from(2) {
            k = &n - &k;
        }

        if k == 0 {
            return Integer::ONE;
        }

        let mut result = Integer::ONE;
        for i in integer_increasing_inclusive_range(Integer::ONE, k.clone()) {
            result *= &n - &k + &i;
            result /= &i;
        }
        result
    }

    fn multinom(k: &[Self]) -> Self {
        let mut result = Integer::ONE;
        let mut acc = Integer::ZERO;

        for value in k {
            acc += value;
            result *= Self::binom(acc.clone(), value.clone())
        }

        result
    }
}
