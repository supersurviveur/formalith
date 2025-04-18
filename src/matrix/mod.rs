//! Matrix implementation.

use std::fmt::Display;

use crate::{
    field::{Group, Ring},
    printer::{Print, PrintOptions},
};

/// A matrix with coefficients living in `T`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Group> {
    size: (usize, usize),
    pub(crate) data: Vec<T::Element>,
    ring: &'static T,
}

impl<T: Group> Matrix<T> {
    /// Create a new matrix
    pub fn new(size: (usize, usize), data: Vec<T::Element>, ring: &'static T) -> Self {
        Self { size, data, ring }
    }
}

impl<T: Group> std::ops::Add<&Self> for Matrix<T> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        for (i, v) in rhs.data.iter().enumerate() {
            self.ring.add_assign(&mut self.data[i], v);
        }
        self
    }
}

impl<T: Group> std::ops::Add for Matrix<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: Group> std::ops::Add for &Matrix<T> {
    type Output = Matrix<T>;

    fn add(self, rhs: Self) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: Group> std::ops::Neg for &Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        let mut res = self.clone();
        for v in &mut res.data {
            self.ring.neg_assign(v);
        }
        res
    }
}

impl<T: Group> std::ops::Neg for Matrix<T> {
    type Output = Matrix<T>;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: Group> PartialOrd for Matrix<T> {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        // There is no order over matrix
        None
    }
}

impl<T: Group> Print for Matrix<T> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::group_delim("[", options, f)?;
        for i in 0..self.size.0 {
            Self::group_delim("[", options, f)?;
            for j in i * self.size.0..(i + 1) * self.size.0 {
                write!(f, "{}", self.data[j])?;
                if j != (i + 1) * self.size.0 - 1 {
                    Self::delimiter(",", options, f)?;
                }
            }
            Self::group_delim("]", options, f)?;
            if i != self.size.0 - 1 {
                Self::delimiter(",", options, f)?;
            }
        }
        Self::group_delim("]", options, f)?;
        Ok(())
    }

    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        todo!()
    }
}

impl<T: Group> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Print::fmt(self, &PrintOptions::default(), f)
    }
}

impl<T: Ring> Matrix<T> {
    /// Réalise l'élimination de Gauss pour mettre la matrice sous forme échelonnée
    pub fn gaussian_elimination(&mut self) {
        let mut det = self.ring.one();
        let (rows, cols) = self.size;
        let mut pivot_row = 0;

        // Parcours des colonnes pivots
        for pivot_col in 0..cols {
            if pivot_row >= rows {
                break;
            }

            // Trouver le pivot non nul
            let pivot = match (pivot_row..rows)
                .find(|&i| !self.ring.is_zero(&self.data[i * cols + pivot_col]))
            {
                Some(p) => p,
                None => continue,
            };

            // Échanger les lignes si nécessaire
            if pivot != pivot_row {
                det = self.ring.inv(&det).unwrap();
                self.swap_rows(pivot_row, pivot);
            }

            // Normaliser la ligne pivot
            let pivot_val = self.data[pivot_row * cols + pivot_col].clone();
            let inv_pivot = self.ring.inv(&pivot_val).expect("Element non inversible");
            self.ring.mul_assign(&mut det, &pivot_val);
            self.scale_row(pivot_row, &inv_pivot);

            // Élimination des éléments sous le pivot
            for row in (pivot_row + 1)..rows {
                let factor = self.data[row * cols + pivot_col].clone();
                if !self.ring.is_zero(&factor) {
                    let neg_factor = self.ring.neg(&factor);
                    self.row_add(row, pivot_row, &neg_factor);
                }
            }

            pivot_row += 1;
        }
        println!("det: {}", det);
    }

    // Méthodes auxiliaires

    /// Échange deux lignes de la matrice
    fn swap_rows(&mut self, row1: usize, row2: usize) {
        let cols = self.size.1;
        let start1 = row1 * cols;
        let start2 = row2 * cols;
        for i in 0..cols {
            self.data.swap(start1 + i, start2 + i);
        }
    }

    /// Multiplie une ligne par un scalaire
    fn scale_row(&mut self, row: usize, factor: &T::Element) {
        let cols = self.size.1;
        for col in 0..cols {
            let index = row * cols + col;
            self.data[index] = self.ring.mul(&self.data[index], factor);
        }
    }

    /// Ajoute à une ligne un multiple d'une autre ligne
    fn row_add(&mut self, target_row: usize, source_row: usize, factor: &T::Element) {
        let cols = self.size.1;
        for col in 0..cols {
            let src_idx = source_row * cols + col;
            let tgt_idx = target_row * cols + col;
            let term = self.ring.mul(&self.data[src_idx], factor);
            self.data[tgt_idx] = self.ring.add(&self.data[tgt_idx], &term);
        }
    }
}
