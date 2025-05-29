//! Implementation of operations over [Term].

use std::cmp::Ordering;

use crate::field::{Ring, RingImpl};

use super::{Add, Mul, Pow, Term, Value};

impl<T: Ring> std::ops::Add for &Term<T> {
    type Output = Term<T>;

    fn add(self, rhs: Self) -> Self::Output {
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
                            }
                        }
                        Err(i) => a1_clone.terms.insert(i, rhs.clone()),
                    }
                }
                Term::Add(a1_clone)
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

impl<T: Ring> std::ops::AddAssign<&Self> for Term<T> {
    fn add_assign(&mut self, rhs: &Self) {
        *self = &*self + &rhs;
    }
}

impl<T: Ring> std::ops::AddAssign for Term<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = &*self + &rhs;
    }
}

impl<T: Ring> std::ops::Add<&Self> for Term<T> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<T: Ring> std::ops::Add for Term<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<T: Ring> std::ops::Mul for &Term<T> {
    type Output = Term<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Term::Mul(Mul::new(vec![self.clone(), rhs.clone()], self.get_set())).normalize()
        // TODO, normalize mul directly like add
    }
}

impl<T: Ring> std::ops::MulAssign<&Self> for Term<T> {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = &*self * &rhs;
    }
}

impl<T: Ring> std::ops::MulAssign for Term<T> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = &*self * &rhs;
    }
}

impl<T: Ring> std::ops::Mul<&Self> for Term<T> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<T: Ring> std::ops::Mul for Term<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<T: Ring> std::ops::Div for &Term<T> {
    type Output = Term<T>;

    fn div(self, rhs: Self) -> Self::Output {
        self * &rhs
            .pow(&Term::Value(Value::new(
                self.get_set().get_exposant_set().nth(-1),
                self.get_set().get_exposant_set(),
            )))
            .normalize()
    }
}

impl<T: Ring> std::ops::DivAssign for Term<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = &*self / &rhs;
    }
}

impl<T: Ring> std::ops::Div<&Self> for Term<T> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        &self / rhs
    }
}

impl<T: Ring> std::ops::Div for Term<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

impl<T: Ring> std::ops::Neg for Term<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: Ring> std::ops::Neg for &Term<T> {
    type Output = Term<T>;

    fn neg(self) -> Self::Output {
        let mut new = self.clone();
        match new {
            Term::Value(ref mut value) => value.ring.neg_assign(&mut value.value),
            Term::Add(ref add) => {
                let ring = add.ring;
                new = Term::Mul(Mul::new(
                    vec![new, Term::Value(Value::new(ring.neg(&ring.one()), ring))],
                    ring,
                ))
            }
            Term::Symbol(ref symbol) => {
                let ring = symbol.ring;
                new = Term::Mul(Mul::new(
                    vec![new, Term::Value(Value::new(ring.neg(&ring.one()), ring))],
                    ring,
                ))
            }
            Term::Mul(ref mut mul) => {
                let coeff = mul.get_coeff();
                mul.set_coeff(mul.ring.neg(&coeff));
            }
            Term::Pow(ref pow) => {
                let ring = pow.set;
                new = Term::Mul(Mul::new(
                    vec![new, Term::Value(Value::new(ring.neg(&ring.one()), ring))],
                    ring,
                ))
            }
            Term::Fun(ref fun) => {
                let set = fun.get_set();
                new = Term::Mul(Mul::new(
                    vec![new, Term::Value(Value::new(set.neg(&set.one()), set))],
                    set,
                ))
            }
        }
        new
    }
}

impl<T: Ring> std::ops::Sub for &Term<T> {
    type Output = Term<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &-rhs
    }
}

impl<T: Ring> std::ops::SubAssign for Term<T> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = &*self - &rhs;
    }
}

impl<T: Ring> std::ops::Sub<&Self> for Term<T> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        &self - rhs
    }
}

impl<T: Ring> std::ops::Sub for Term<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        &self - &rhs
    }
}

impl<T: Ring> Term<T> {
    /// Compute `self^exposant`
    pub fn pow(&self, exposant: &Term<T::ExposantSet>) -> Term<T> {
        let res = Term::Pow(Pow::new(self.clone(), exposant.clone(), self.get_set()));
        res.normalize()
    }
}
