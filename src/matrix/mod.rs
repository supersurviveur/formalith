use std::fmt::Display;

use crate::field::{Group, Ring};

#[derive(Clone, Debug, PartialEq)]
pub struct Matrix<T: Group> {
    size: (usize, usize),
    pub(crate) data: Vec<T::Element>,
    ring: &'static T,
}

impl<T: Group> Matrix<T> {
    pub fn new(size: (usize, usize), data: Vec<T::Element>, ring: &'static T) -> Self {
        Self { size, data, ring }
    }
}

impl<T: Group> PartialOrd for Matrix<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl<T: Group> Display for Matrix<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for i in 0..self.size.0 {
            write!(f, "[")?;
            for j in i * self.size.0..(i + 1) * self.size.0 {
                write!(f, "{}", self.data[j])?;
                if j != (i + 1) * self.size.0 - 1 {
                    write!(f, ",")?;
                }
            }
            write!(f, "]")?;
            if i != self.size.0 - 1 {
                write!(f, ",")?;
            }
        }
        write!(f, "]")?;
        Ok(())
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
