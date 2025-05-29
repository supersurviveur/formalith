//! Polynom implementation

use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
};

use crate::{
    field::{Group, Ring},
    printer::{PrettyPrinter, Print, PrintOptions},
    term::Term,
};

pub trait Monomial: Debug + Clone + PartialEq + Eq + Hash + PartialOrd + Print {}

impl<T: Debug + Clone + PartialEq + Eq + Hash + PartialOrd + Print> Monomial for T {}

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct MultivariatePolynomial<V: Monomial, T: Group, U: Group> {
    terms: Vec<(Vec<(V, U::Element)>, T::Element)>,
    set: T,
    exposant_set: U,
}

impl<V: Monomial, T: Ring, U: Ring> MultivariatePolynomial<V, T, U> {
    pub fn new(terms: Vec<(Vec<(V, U::Element)>, T::Element)>, set: T, exposant_set: U) -> Self {
        Self {
            terms,
            set,
            exposant_set,
        }
    }
    /// Check if the polynom is a constant
    pub fn is_constant(&self) -> bool {
        self.terms.len() == 1 && self.terms[0].0.is_empty()
    }

    pub fn constant(coeff: T::Element, set: T, exposant_set: U) -> Self {
        Self {
            terms: vec![(vec![], coeff)],
            set,
            exposant_set,
        }
    }

    pub fn variable(var: V, set: T, exposant_set: U) -> Self {
        Self {
            terms: vec![(vec![(var, exposant_set.one())], set.one())],
            set,
            exposant_set,
        }
    }

    /// Check if the polynom is the constant 1
    pub fn is_one(&self) -> bool {
        self.is_constant() && self.set.is_one(&self.terms[0].1)
    }

    pub fn one(set: T, exposant_set: U) -> Self {
        Self {
            terms: vec![(vec![], set.one())],
            set,
            exposant_set,
        }
    }

    /// Check if the polynom is the constant 1
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.iter().all(|(_, c)| self.set.is_zero(c))
    }

    pub fn zero(set: T, exposant_set: U) -> Self {
        Self {
            terms: vec![(vec![], set.zero())],
            set,
            exposant_set,
        }
    }

    fn combine_vars(
        vars1: &[(V, U::Element)],
        vars2: &[(V, U::Element)],
        exposant_set: U,
    ) -> Vec<(V, U::Element)> {
        let mut combined = HashMap::new();

        for (v, e) in vars1.iter().chain(vars2.iter()) {
            *combined.entry(v.clone()).or_insert(exposant_set.zero()) =
                exposant_set.add(combined.get(v).unwrap_or(&exposant_set.zero()), e);
        }

        let mut sorted: Vec<_> = combined.into_iter().collect();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Separate the polynom on a variable
    pub fn separate_variable(&self, var: &V) -> (Vec<Self>, Vec<U::Element>) {
        let mut coefficients = Vec::new();
        let mut exponents = Vec::new();

        for (vars, coeff) in &self.terms {
            let mut var_exponents = vars
                .iter()
                .filter(|(v, _)| v == var)
                .map(|(_, e)| e.clone())
                .collect::<Vec<_>>();

            let exp = var_exponents.pop().unwrap_or(self.exposant_set.zero());
            let remaining_vars = vars.iter().filter(|(v, _)| v != var).cloned().collect();

            coefficients.push(Self {
                terms: vec![(remaining_vars, coeff.clone())],
                set: self.set,
                exposant_set: self.exposant_set,
            });
            exponents.push(exp);
        }

        (coefficients, exponents)
    }

    /// Get the leading monomial
    pub fn leading_term(&self) -> (Vec<(V, U::Element)>, T::Element) {
        self.terms.first().cloned().expect("Aucun terme")
    }

    /// Convert the polynomial to an expression
    pub fn to_term(&self) -> Term<T>
    where
        Term<T>: From<V>,
        Term<T::ExposantSet>: From<U::Element>,
    {
        let mut res = Term::zero(self.set);
        for monomial in &self.terms {
            let mut term = Term::one(self.set);
            for (var, exp) in &monomial.0 {
                term *= Term::from(var.clone()).pow(&Term::from(exp.clone()));
            }
            term *= Term::constant(monomial.1.clone(), self.set);
            res += term;
        }
        res
    }
}

impl<V: Monomial, T: Ring, U: Ring> MultivariatePolynomial<V, T, U> {}

impl<V: Monomial, T: Ring, U: Ring> MultivariatePolynomial<V, T, U>
where
    T::Element: TryFrom<U::Element>,
    V: TryFrom<U::Element>,
    <V as TryFrom<U::Element>>::Error: Debug,
{
    fn derivative(&self, var: &V) -> Self {
        let mut terms = Vec::new();

        for (vars, coeff) in &self.terms {
            let mut new_vars = vars.clone();

            for (i, (v, exp)) in new_vars.iter_mut().enumerate() {
                if v == var {
                    let new_exp = self.exposant_set.sub(exp, &self.exposant_set.one());
                    let new_coeff = if let Ok(new_coeff) =
                        T::Element::try_from(exp.clone()).map(|c| self.set.mul(coeff, &c))
                    {
                        new_coeff
                    } else {
                        // Convert exponent to variable
                        terms.push((
                            vec![(
                                V::try_from(exp.clone())
                                    .expect("Exponent must be convertible to T or V"),
                                self.exposant_set.one(),
                            )],
                            self.set.one(),
                        ));
                        coeff.clone()
                    };

                    if self.exposant_set.is_zero(&new_exp) {
                        new_vars.remove(i);
                    } else {
                        *exp = new_exp;
                    }

                    terms.push((new_vars, new_coeff));
                    break;
                }
            }
        }

        Self {
            terms,
            set: self.set,
            exposant_set: self.exposant_set,
        }
    }
}

impl<V: Monomial, T: Ring, U: Ring> MultivariatePolynomial<V, T, U>
where
    U::Element: Ord,
{
    /// Compute the degree of the polynomial.
    pub fn degree(&self) -> U::Element {
        self.terms
            .iter()
            .map(|(vars, _)| {
                vars.iter()
                    .map(|(_, e)| e.clone())
                    .fold(self.exposant_set.zero(), |a, b| {
                        self.exposant_set.add(&a, &b)
                    })
            })
            .max()
            .unwrap_or(self.exposant_set.zero())
    }
}
impl<V: Monomial, T: Ring, U: Ring> MultivariatePolynomial<V, T, U> {
    /// Create a polynom from a hashmap which associate monomial to its coefficient
    fn from_term_map(
        map: HashMap<Vec<(V, U::Element)>, T::Element>,
        set: T,
        exposant_set: U,
    ) -> Self {
        let mut res = Self {
            terms: map.into_iter().collect(),
            set,
            exposant_set,
        };
        MultivariatePolynomial::normalize(&mut res);
        res
    }
    fn from_term(term: (Vec<(V, U::Element)>, T::Element), set: T, exposant_set: U) -> Self {
        let mut res = Self {
            terms: vec![term],
            set,
            exposant_set,
        };
        MultivariatePolynomial::normalize(&mut res);
        res
    }

    /// Normalize the polynomial.
    pub fn normalize(&mut self) {
        // Remove monomials
        self.terms.retain(|(_, coeff)| !self.set.is_zero(coeff));
        // Sort variables in monomials
        self.terms.iter_mut().for_each(|monomial| {
            // Remove variables
            monomial.0.retain(|(_, e)| !self.exposant_set.is_zero(e));
            monomial
                .0
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        });
        // Sort monomials
        self.terms.sort_by(|a, b| {
            b.0.iter()
                .map(|(_, e)| e.clone())
                .partial_cmp(a.0.iter().map(|(_, e)| e.clone()))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Euclidian division of a monomial by another
    fn quot_rem_monomial(
        num_vars: &[(V, U::Element)],
        num_coeff: &T::Element,
        denom_vars: &[(V, U::Element)],
        denom_coeff: &T::Element,
        set: T,
        exposant_set: U,
    ) -> Option<(Vec<(V, U::Element)>, T::Element)> {
        let denom_inv = set.try_inv(denom_coeff).unwrap();
        let q_coeff = set.mul(num_coeff, &denom_inv);

        let mut var_map: HashMap<_, _> = num_vars.iter().cloned().collect();
        for (v, e) in denom_vars {
            match var_map.get_mut(v) {
                Some(current_exp) => {
                    if *current_exp < *e {
                        return None;
                    }
                    *current_exp = exposant_set.sub(current_exp, e);
                    if exposant_set.is_zero(current_exp) {
                        var_map.remove(v);
                    }
                }
                None => return None,
            }
        }

        let mut q_vars: Vec<_> = var_map.into_iter().collect();
        q_vars.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Some((q_vars, q_coeff))
    }

    /// Euclidian division, returning quotient and remainder
    pub fn quot_rem(&self, divisor: &Self) -> (Self, Self) {
        let mut quotient = Self::zero(self.set, self.exposant_set);
        let mut remainder = Self::zero(self.set, self.exposant_set);
        let mut current_dividend = self.clone();

        while !current_dividend.is_zero() {
            let (lead_vars, lead_coeff) = current_dividend.leading_term();
            let (div_vars, div_coeff) = divisor.leading_term();

            match Self::quot_rem_monomial(
                &lead_vars,
                &lead_coeff,
                &div_vars,
                &div_coeff,
                self.set,
                self.exposant_set,
            ) {
                Some((q_vars, q_coeff)) => {
                    let term_poly = Self::from_term((q_vars, q_coeff), self.set, self.exposant_set);

                    quotient = &quotient + &term_poly;
                    let to_subtract = &term_poly * &divisor;
                    current_dividend = &current_dividend - &to_subtract;
                }
                None => {
                    let term =
                        Self::from_term((lead_vars, lead_coeff), self.set, self.exposant_set);
                    remainder = remainder + &term;
                    current_dividend = &current_dividend - &term;
                }
            }
        }

        (quotient, remainder)
    }
    /// Compute GCD between self and other using euclide algorithm
    /// TODO divide by coefficient of higher exposant to get a unitary polynomial
    pub fn gcd(&self, other: &Self) -> Self {
        if other.is_zero() {
            return self.clone();
        }

        let (_, remainder) = self.quot_rem(other);
        let mut res = other.gcd(&remainder);
        let leading_coeff = res.leading_term().1;
        if !self.set.is_one(&leading_coeff) {
            res = &res
                * &Self::constant(
                    self.set.try_inv(&leading_coeff).unwrap(),
                    self.set,
                    self.exposant_set,
                );
        }
        res
    }
    /// Compute the content of a polynom
    pub fn content(&self, var: &V) -> Self {
        let (coeffs, _) = self.separate_variable(var);
        coeffs
            .into_iter()
            .fold(Self::one(self.set, self.exposant_set), |acc, x| acc.gcd(&x))
    }
    /// Primitive part of the polynomial
    pub fn primitive_part(&self, var: &V) -> Self {
        let content = self.content(var);
        self.quot_rem(&content).0
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Add for &MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn add(self, rhs: Self) -> Self::Output {
        // Polynomial must be already normalized
        let mut term_map = HashMap::new();

        for (vars, coeff) in &self.terms {
            *term_map.entry(vars.clone()).or_insert(self.set.zero()) = self
                .set
                .add(term_map.get(vars).unwrap_or(&self.set.zero()), coeff);
        }

        for (vars, coeff) in &rhs.terms {
            *term_map.entry(vars.clone()).or_insert(self.set.zero()) = self
                .set
                .add(term_map.get(vars).unwrap_or(&self.set.zero()), coeff);
        }

        MultivariatePolynomial::from_term_map(term_map, self.set, self.exposant_set)
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Add for MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Add<&Self> for MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn add(self, rhs: &Self) -> Self::Output {
        &self + rhs
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Mul for &MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut term_map = HashMap::new();

        for (vars1, coeff1) in &self.terms {
            for (vars2, coeff2) in &rhs.terms {
                // Fusion des variables avec addition des exposants
                let combined_vars = MultivariatePolynomial::<V, T, U>::combine_vars(
                    vars1,
                    vars2,
                    self.exposant_set,
                );
                let product_coeff = self.set.mul(coeff1, coeff2);

                *term_map.entry(combined_vars).or_insert(self.set.zero()) = self.set.add(
                    term_map.get(&combined_vars).unwrap_or(&self.set.zero()),
                    &product_coeff,
                );
            }
        }

        MultivariatePolynomial::<V, T, U>::from_term_map(term_map, self.set, self.exposant_set)
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Mul for MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Mul<&Self> for MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn mul(self, rhs: &Self) -> Self::Output {
        &self * rhs
    }
}

impl<V: Monomial, T: Ring, U: Ring> std::ops::Sub for &MultivariatePolynomial<V, T, U> {
    type Output = MultivariatePolynomial<V, T, U>;

    fn sub(self, rhs: Self) -> Self::Output {
        let mut term_map = HashMap::new();

        // Ajouter les termes du premier polynôme
        for (vars, coeff) in &self.terms {
            *term_map.entry(vars.clone()).or_insert(self.set.zero()) = self
                .set
                .add(term_map.get(vars).unwrap_or(&self.set.zero()), coeff);
        }

        // Soustraire les termes du second polynôme
        for (vars, coeff) in &rhs.terms {
            *term_map.entry(vars.clone()).or_insert(self.set.zero()) = self
                .set
                .sub(term_map.get(vars).unwrap_or(&self.set.zero()), coeff);
        }

        MultivariatePolynomial::<V, T, U>::from_term_map(term_map, self.set, self.exposant_set)
    }
}

impl<V: Monomial, T: Ring, U: Ring> MultivariatePolynomial<V, T, U>
where
    T::Element: TryFrom<U::Element>,
    V: TryFrom<U::Element>,
    <V as TryFrom<U::Element>>::Error: Debug,
{
    pub fn factor(&self) -> Vec<(Self, u32)> {
        self.square_free_factorization()
        // TODO add cantor-zassenhaus or/and tragger algorithm
    }
    pub fn square_free_factorization(&self) -> Vec<(Self, u32)> {
        let mut factors = Vec::new();

        if self.is_constant() {
            return vec![(self.clone(), 1)];
        }

        let main_var = self.choose_main_variable();

        let content = self.content(&main_var);
        let primitive_part = self.primitive_part(&main_var);

        for (factor, power) in content.square_free_factorization() {
            factors.push((factor, power));
        }

        let derivative = primitive_part.derivative(&main_var);
        let gcd = primitive_part.gcd(&derivative);

        let (mut current, _) = self.quot_rem(&gcd);
        let (mut current_derivative, _) = derivative.quot_rem(&gcd);
        let mut multiplicity = 1;

        while !current.is_constant() {
            let tmp = &current_derivative - &current.derivative(&main_var);
            let g = current.gcd(&tmp);
            current = current.quot_rem(&g).0;
            current_derivative = tmp.quot_rem(&g).0;

            if !g.is_one() {
                factors.push((g, multiplicity));
            }

            multiplicity += 1;
        }

        factors.retain(|(elem, _)| !elem.is_one());

        factors
    }

    fn choose_main_variable(&self) -> V {
        let mut var_counts = HashMap::new();
        for (vars, _) in &self.terms {
            for (v, e) in vars {
                if !self.exposant_set.is_zero(e) {
                    *var_counts.entry(v.clone()).or_insert(0) += 1;
                }
            }
        }
        var_counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(v, _)| v)
            .expect("Polynomial must contain at least one variable")
    }
}

impl<V: Monomial + Print, T: Ring, U: Ring> Print for MultivariatePolynomial<V, T, U> {
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }

    fn pretty_print(&self, options: &PrintOptions) -> PrettyPrinter {
        let mut res = PrettyPrinter::empty();
        for (vars, coeff) in self.terms.iter() {
            let has_coeff = !self.set.is_one(coeff) || vars.is_empty();
            let mut coeff = if has_coeff {
                self.set.pretty_print(coeff, options)
            } else {
                PrettyPrinter::empty()
            };

            for (i, (var, exposant)) in vars.iter().enumerate() {
                let mut var = var.pretty_print(options);
                if !self.exposant_set.is_one(exposant) {
                    var.pow(&self.exposant_set.pretty_print(exposant, options));
                }
                coeff.concat(if i == 0 && !has_coeff { "" } else { "⋅" }, false, &var);
            }
            res.concat("+", true, &coeff);
        }
        res
    }
}

impl<V: Monomial + Print, T: Ring, U: Ring> Display for MultivariatePolynomial<V, T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}
