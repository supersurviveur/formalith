//! Implementation of operations over [Term].

use std::cmp::Ordering;

use super::{Add, Mul, Pow, Term, Value};
use crate::{
    field::{Group, RingBound},
    term::{MergeTerm, flags::Flags},
};

impl<T: RingBound> std::ops::Add for &Term<T> {
    type Output = Term<T>;

    fn add(self, rhs: Self) -> Self::Output {
        // trace!(
        //     "Compute the addition between {} and {}",
        //     self.stdout(),
        //     rhs.stdout()
        // );
        debug_assert!(!self.needs_normalization(), "{:#?}", self);
        debug_assert!(!rhs.needs_normalization());
        let ring = self.get_set();
        if let Term::Add(a1) = self {
            if let Term::Add(a2) = rhs {
                let mut res = Add::with_capacity(a1.len() + a2.len(), ring);
                let mut a1_iter = a1.terms.iter();
                let mut a2_iter = a2.terms.iter();
                let mut a1_cursor = a1_iter.next();
                let mut a2_cursor = a2_iter.next();
                while a1_cursor.is_some() || a2_cursor.is_some() {
                    if let Some(a1) = a1_cursor {
                        if let Some(a2) = a2_cursor {
                            match a1.cmp_terms(a2) {
                                std::cmp::Ordering::Less => {
                                    res.push(a1.clone());
                                    a2_cursor = Some(a2); // Give the borrow back to cursor
                                    a1_cursor = a1_iter.next();
                                }
                                std::cmp::Ordering::Equal => {
                                    let mut a1_clone = a1.clone();
                                    if a1_clone.merge_terms(a2) {
                                        res.push(a1_clone);
                                    } else {
                                        res.push(a1_clone);
                                        res.push(a2.clone());
                                    }
                                    a1_cursor = a1_iter.next();
                                    a2_cursor = a2_iter.next();
                                }
                                std::cmp::Ordering::Greater => {
                                    res.push(a2.clone());
                                    a1_cursor = Some(a1); // Give the borrow back to cursor
                                    a2_cursor = a2_iter.next()
                                }
                            }
                        } else {
                            res.push(a1.clone());
                            a1_cursor = a1_iter.next()
                        }
                    } else if let Some(a2) = a2_cursor {
                        res.push(a2.clone());
                        a2_cursor = a2_iter.next()
                    }
                }
                res.set_normalized(true);
                Term::Add(res)
            } else {
                let mut a1_clone = a1.clone();
                if !rhs.is_zero() {
                    match a1_clone
                        .terms
                        .binary_search_by(|value| value.cmp_terms(rhs))
                    {
                        Ok(i) => {
                            if !a1_clone.terms[i].merge_terms(rhs) {
                                a1_clone.terms.insert(i, rhs.clone())
                            } else if a1_clone.terms[i].is_zero() {
                                a1_clone.terms.remove(i);
                            }
                        }
                        Err(i) => a1_clone.terms.insert(i, rhs.clone()),
                    }
                }
                if a1_clone.len() == 1 {
                    a1_clone.terms[0].clone()
                } else {
                    Term::Add(a1_clone)
                }
            }
        } else if let Term::Add(_) = rhs {
            rhs.add(self)
        } else {
            let mut self_clone = self.clone();
            if self_clone.merge_terms(rhs) {
                self_clone
            } else {
                Term::Add(Add::new(
                    if self_clone.cmp_terms(rhs) == Ordering::Greater {
                        vec![rhs.clone(), self_clone]
                    } else {
                        vec![self_clone, rhs.clone()]
                    },
                    ring,
                ))
                .normalize()
            }
        }
    }
}

impl<T: RingBound> std::ops::AddAssign<&Self> for Term<T> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = &*self + rhs;
    }
}

impl<T: RingBound> std::ops::AddAssign for Term<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl<T: RingBound> std::ops::Add<&Self> for Term<T> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<T: RingBound> std::ops::Add for Term<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T: RingBound> std::ops::Mul for &Term<T> {
    type Output = Term<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(!self.needs_normalization());
        debug_assert!(!rhs.needs_normalization());
        Term::Mul(Mul::new(
            self.get_set().get_coefficient_set().nth(1),
            vec![self.clone(), rhs.clone()],
            self.get_set(),
        ))
        .normalize()
        // TODO, normalize mul directly like add
    }
}

impl<T: RingBound> std::ops::MulAssign<&Self> for Term<T> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = &*self * rhs;
    }
}

impl<T: RingBound> std::ops::MulAssign for Term<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = &*self * &rhs;
    }
}

impl<T: RingBound> std::ops::Mul<&Self> for Term<T> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<T: RingBound> std::ops::Mul for Term<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<T: RingBound> std::ops::Div for &Term<T> {
    type Output = Term<T>;

    fn div(self, rhs: Self) -> Self::Output {
        debug_assert!(!self.needs_normalization());
        debug_assert!(!rhs.needs_normalization());
        self * &rhs
            .pow(&Term::Value(Value::new(
                self.get_set().get_exposant_set().nth(-1),
                self.get_set().get_exposant_set(),
            )))
            .normalize()
    }
}

impl<T: RingBound> std::ops::DivAssign for Term<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = &*self / &rhs;
    }
}

impl<T: RingBound> std::ops::Div<&Self> for Term<T> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}

impl<T: RingBound> std::ops::Div for Term<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T: RingBound> std::ops::Neg for Term<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: RingBound> std::ops::Neg for &Term<T> {
    type Output = Term<T>;

    fn neg(self) -> Self::Output {
        debug_assert!(!self.needs_normalization());
        let mut new = self.clone();
        match new {
            Term::Value(ref mut value) => value.set.neg_assign(&mut value.value),
            Term::Add(ref add) => {
                let ring = add.set;
                new = Term::Mul(Mul::new(
                    ring.get_coefficient_set().nth(-1),
                    vec![new],
                    ring,
                ))
                .normalize();
            }
            Term::Symbol(ref symbol) => {
                let ring = symbol.ring;
                new = Term::Mul(Mul::new(
                    ring.get_coefficient_set().nth(-1),
                    vec![new],
                    ring,
                ))
                .normalize()
            }
            Term::Mul(ref mut mul) => {
                let coeff = mul.get_coeff();
                mul.set_coeff(mul.set.get_coefficient_set().neg(&coeff));
            }
            Term::Pow(ref pow) => {
                let ring = pow.set;
                new = Term::Mul(Mul::new(
                    ring.get_coefficient_set().nth(-1),
                    vec![new],
                    ring,
                ))
                .normalize()
            }
            Term::Fun(ref fun) => {
                let set = fun.get_set();
                new = Term::Mul(Mul::new(set.get_coefficient_set().nth(-1), vec![new], set))
                    .normalize()
            }
        }
        new
    }
}

impl<T: RingBound> std::ops::Sub for &Term<T> {
    type Output = Term<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(!self.needs_normalization());
        debug_assert!(!rhs.needs_normalization());
        self + &-rhs
    }
}

impl<T: RingBound> std::ops::SubAssign for Term<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<T: RingBound> std::ops::Sub<&Self> for Term<T> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl<T: RingBound> std::ops::Sub for Term<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T: RingBound> Term<T> {
    /// Compute `self^exposant`
    pub fn pow(&self, exposant: &Term<T::ExposantSet>) -> Term<T> {
        debug_assert!(!self.needs_normalization());
        debug_assert!(!exposant.needs_normalization());
        let res = Term::Pow(Pow::new(self.clone(), exposant.clone(), self.get_set()));
        res.normalize()
    }
}
