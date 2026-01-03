//! Matrix implementation.

use std::{
    fmt::Display,
    ops::{Index, IndexMut},
};

use crate::{
    field::{Group, Ring, Set},
    printer::{PrettyPrint, PrettyPrinter, Print, PrintOptions},
};

/// A matrix with coefficients living in `T`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Matrix<T: Set> {
    size: (usize, usize),
    pub(crate) data: Vec<T::Element>,
    set: T,
}

/// An error thrown by a matrix method.
#[derive(Debug)]
pub enum MatrixError {
    /// The matrix is not a square matrix.
    NotSquare,
    /// A pivot in a row reduction is not invertible.
    NonInvertiblePivot,
}

/// Result type of matrix methods.
pub type MatrixResult<T> = Result<T, MatrixError>;

impl<T: Set> Matrix<T> {
    /// Create a new matrix
    pub const fn new(size: (usize, usize), data: Vec<T::Element>, set: T) -> Self {
        Self { size, data, set }
    }

    /// Get the size of the matrix (height * width)
    pub const fn size(&self) -> usize {
        self.width() * self.height()
    }

    /// Get the width of the matrix.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.size.1
    }

    /// Get the height of the matrix.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.size.0
    }

    /// Swap two rows in the matrix.
    pub fn swap_rows(&mut self, row1: usize, row2: usize) {
        let cols = self.size.1;
        let start1 = row1 * cols;
        let start2 = row2 * cols;
        for i in 0..cols {
            self.data.swap(start1 + i, start2 + i);
        }
    }
}

impl<T: Group> Matrix<T> {
    /// Create a new matrix filled with zeros.
    pub fn zero(size: (usize, usize), set: T) -> Self {
        Self::new(size, vec![set.zero(); size.0 * size.1], set)
    }
}

impl<T: Group> std::ops::Add<&Self> for Matrix<T> {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        for (i, v) in rhs.data.iter().enumerate() {
            self.set.add_assign(&mut self.data[i], v);
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
            self.set.neg_assign(v);
        }
        res
    }
}

impl<T: Group> std::ops::Neg for Matrix<T> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        -&self
    }
}

impl<T: Ring> std::ops::Mul for Matrix<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}
impl<T: Ring> std::ops::Mul for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, _rhs: Self) -> Self::Output {
        todo!()
    }
}

impl<T: Set> PartialOrd for Matrix<T> {
    fn partial_cmp(&self, _: &Self) -> Option<std::cmp::Ordering> {
        // There is no order over matrix
        None
    }
}

impl<T: Set> Print for Matrix<T> {
    fn print(
        &self,
        options: &crate::printer::PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::group_delim("[", options, f)?;
        for i in 0..self.size.0 {
            Self::group_delim("[", options, f)?;
            for j in i * self.size.0..(i + 1) * self.size.0 {
                self.set.print(&self.data[j], options, f)?;
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
}
impl<T: Set> PrettyPrint for Matrix<T> {
    fn pretty_print(&self, options: &PrintOptions) -> crate::printer::PrettyPrinter {
        let mut coeffs: Vec<_> = self
            .data
            .iter()
            .map(|x| self.set.pretty_print(x, options))
            .collect();
        let mut lines_height = vec![0; self.size.0];
        let mut lines_baselines = vec![0; self.size.0];
        let mut columns_width = vec![0; self.size.1];
        for line in 0..self.size.0 {
            lines_height[line] = coeffs[line * self.size.1..(line + 1) * self.size.1]
                .iter()
                .max_by_key(|x| x.height())
                .unwrap()
                .height();
            lines_baselines[line] = coeffs[line * self.size.1..(line + 1) * self.size.1]
                .iter()
                .max_by_key(|x| x.baseline)
                .unwrap()
                .baseline;
        }
        for column in 0..self.size.1 {
            columns_width[column] = coeffs[column..self.size()]
                .iter()
                .step_by(self.size.1)
                .max_by_key(|x| x.width())
                .unwrap()
                .width();
        }
        for x in 0..self.height() {
            for y in 0..self.width() {
                lines_height[x] = coeffs[x * self.width() + y].center(
                    lines_height[x],
                    columns_width[y],
                    lines_baselines[x],
                );
            }
        }
        let mut res = PrettyPrinter::empty();
        for i in 0..self.size.1 {
            let mut column = PrettyPrinter::empty();
            for coeff in coeffs[i..self.size()].iter().step_by(self.size.1) {
                column.vertical_concat(" ", coeff);
            }
            res.concat("", true, &column);
        }
        res.group('[', ']');
        res.baseline = res.height() / 2;
        res
    }
}

impl<T: Set> Display for Matrix<T> {
    default fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        PrettyPrint::fmt(self, &PrintOptions::default(), f)
    }
}

impl<T: Ring> Matrix<T> {
    /// Transform the matrix in place in row reduced echelon form.
    ///
    /// # Errors
    /// The method will fail if a pivot is not invertible.
    pub fn row_reduce(&mut self) -> MatrixResult<()> {
        self.partial_row_reduce()?;
        self.back_substitution()
    }
    /// Transform a matrix in row $echelon form to a matrix in row reduced echelon form.
    ///
    /// # Errors
    /// The method will fail if a pivot is not invertible.
    pub fn back_substitution(&mut self) -> MatrixResult<()> {
        for line in (0..self.height()).rev() {
            // Find the pivot
            if let Some(pivot) = (0..self.width()).find(|x| !self.set.is_zero(&self[(line, *x)])) {
                if !self.set.is_one(&self[(line, pivot)]) {
                    self.scale_row(
                        line,
                        &self
                            .set
                            .try_inv(&self[(line, pivot)])
                            .ok_or(MatrixError::NonInvertiblePivot)?,
                    );
                }
                // Remove coefficients under the pivot
                for i in 0..line {
                    self.row_add(i, line, &self.set.neg(&self[(i, pivot)]));
                    // Coefficient under the pivot became zero, force the value to avoid complex expression
                    self[(i, pivot)] = self.set.zero();
                }
            }
        }
        Ok(())
    }

    /// Compute the inverse of the matrix if possible.
    ///
    /// # Errors
    /// The method will fail if the matrix is not square.
    pub fn inv(&self) -> MatrixResult<Self> {
        if self.width() != self.height() {
            return Err(MatrixError::NotSquare);
        }
        let mut augmented = Self::zero((self.size.0, self.size.1 * 2), self.set);
        for line in 0..self.height() {
            for column in 0..self.width() {
                augmented[(line, column)] = self[(line, column)].clone();
            }
            augmented[(line, line + self.width())] = self.set.one();
        }
        augmented.row_reduce()?;

        for line in 0..self.height() {
            for column in 0..self.width() {
                augmented[line * self.width() + column] =
                    augmented[(line, column + self.width())].clone();
            }
        }
        augmented.size = self.size;
        Ok(augmented)
    }
}

impl<T: Ring> Matrix<T> {
    /// Transforn the matrix in place in row echelon form, returning its rank.
    ///
    /// # Errors
    /// The method will fail if a pivot is not inversible.
    pub fn partial_row_reduce(&mut self) -> MatrixResult<usize> {
        let (rows, cols) = self.size;
        let mut pivot_row = 0;

        for pivot_col in 0..cols {
            if pivot_row >= rows {
                break;
            }

            // Find a non-null pivot
            let Some(pivot) =
                (pivot_row..rows).find(|&i| !self.set.is_zero(&self.data[i * cols + pivot_col]))
            else {
                continue;
            };

            if pivot != pivot_row {
                self.swap_rows(pivot_row, pivot);
            }

            // Remove coefficient under the pivot
            for row in (pivot_row + 1)..rows {
                let factor = self.data[row * cols + pivot_col].clone();

                if !self.set.is_zero(&factor) {
                    let pivot_val = self.data[pivot_row * cols + pivot_col].clone();
                    let ratio = self.set.mul(
                        &factor,
                        &self
                            .set
                            .try_inv(&pivot_val)
                            .ok_or(MatrixError::NonInvertiblePivot)?,
                    );
                    let neg_ratio = self.set.neg(&ratio);
                    self.row_add(row, pivot_row, &neg_ratio);
                }
                // Coefficient under the pivot became zero, force the value to avoid complex expression
                self.data[row * cols + pivot_col] = self.set.zero();
            }

            pivot_row += 1;
        }
        Ok(pivot_row)
    }

    /// Compute the determinant of the matrix.
    ///
    /// # Errors
    /// This will returns an error if the matrix is not square.
    pub fn det(&self) -> MatrixResult<T::Element> {
        if self.height() != self.width() {
            return Err(MatrixError::NotSquare);
        }
        match self.height() {
            0 => unreachable!(),
            1 => Ok(self.data[0].clone()),
            2 => Ok(self.set.sub(
                &self.set.mul(&self.data[0], &self.data[3]),
                &self.set.mul(&self.data[1], &self.data[2]),
            )),
            _ => self.clone().det_in_place(),
        }
    }
    /// Compute the determinant of the matrix in place. Matrix will be in partially reduced form then.
    ///
    /// # Errors
    /// This will returns an error if the matrix is not square or if a pivot is not invertible.
    pub fn det_in_place(&mut self) -> MatrixResult<T::Element> {
        if self.height() != self.width() {
            return Err(MatrixError::NotSquare);
        }

        self.partial_row_reduce()?;
        let mut det = self.set.one();
        for line in 0..self.width() {
            self.set
                .mul_assign(&mut det, &self.data[line * self.width() + line]);
        }
        Ok(det)
    }

    /// Scale a row by a factor.
    pub fn scale_row(&mut self, row: usize, factor: &T::Element) {
        let cols = self.size.1;
        for col in 0..cols {
            let index = row * cols + col;
            self.data[index] = self.set.mul(&self.data[index], factor);
        }
    }

    /// Add a row multiplied by a factor to another.
    pub fn row_add(&mut self, target_row: usize, source_row: usize, factor: &T::Element) {
        let cols = self.size.1;
        for col in 0..cols {
            let src_idx = source_row * cols + col;
            let tgt_idx = target_row * cols + col;
            let term = self.set.mul(&self.data[src_idx], factor);
            self.data[tgt_idx] = self.set.add(&self.data[tgt_idx], &term);
        }
    }

    /// Scale the matrix by a factor
    pub fn external_product(&mut self, factor: &T::Element) {
        let rows = self.size.0;
        for row in 0..rows {
            self.scale_row(row, factor);
        }
    }
}

impl<T: Set> Index<usize> for Matrix<T> {
    type Output = T::Element;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Set> IndexMut<usize> for Matrix<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Set> Index<(usize, usize)> for Matrix<T> {
    type Output = T::Element;

    /// Get the `i`th row and `j`th column of the matrix, where `index=(i,j)`.
    #[inline]
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.data[index.0 * self.width() + index.1]
    }
}

impl<T: Set> IndexMut<(usize, usize)> for Matrix<T> {
    /// Get the `i`th row and `j`th column of the matrix, where `index=(i,j)`.
    #[inline]
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let width = self.width();
        &mut self.data[index.0 * width + index.1]
    }
}
