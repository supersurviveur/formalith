//! Printing trait and options to render expressions.

use std::fmt::Display;

use owo_colors::{
    colors::{BrightBlack, Yellow},
    Color, OwoColorize,
};
use phf::phf_map;

use crate::{field::Group, term::Term};

/// Rendering options are stored here. It can be created from some presets or completly custom.
pub struct PrintOptions {
    /// Set color rendering. Enabled by default if stdout support colors.
    pub colors: bool,
    /// Set rendering to typst format. Disabled by default.
    pub typst: bool,
    /// Fancier printing using ASCII art.
    pub pretty_print: bool,
}

impl Default for PrintOptions {
    fn default() -> Self {
        PrintOptions {
            colors: supports_color::on_cached(supports_color::Stream::Stdout).is_some(),
            typst: false,
            pretty_print: true,
        }
    }
}

/// Elements and expressions which can be printed using options.
pub trait Print {
    /// Format `self` using given options, selecting or not pretty printing.
    #[must_use]
    fn fmt(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if options.pretty_print {
            write!(f, "{}", self.pretty_print(options))
        } else {
            self.print(options, f)
        }
    }
    /// Format `self` using given options.
    #[must_use]
    fn print(&self, options: &PrintOptions, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result;
    /// Format `self` using given options, with pretty printing.
    #[must_use]
    fn pretty_print(&self, options: &PrintOptions) -> PrettyPrinter;

    /// Apply delimiter color depending on options.
    #[must_use]
    #[inline(always)]
    fn delimiter(
        text: &str,
        options: &PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::fg::<BrightBlack>(text, options, f)
    }

    /// Add paren to the expr if needed.
    #[must_use]
    #[inline(always)]
    fn group<T: Group>(
        elem: &Term<T>,
        options: &PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let need_paren = match elem {
            Term::Value(_) | Term::Symbol(_) | Term::Fun(_) => false,
            Term::Add(_) | Term::Pow(_) | Term::Mul(_) => true,
        };
        if need_paren {
            Self::group_delim("(", options, f)?;
        }
        write!(f, "{}", elem)?;
        if need_paren {
            Self::group_delim(")", options, f)?;
        }
        Ok(())
    }

    /// Apply group delimiter color depending on options.
    #[must_use]
    #[inline(always)]
    fn group_delim(
        text: &str,
        options: &PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if options.colors {
            write!(f, "{}", text.white().dimmed())?;
        } else {
            write!(f, "{}", text)?;
        }
        Ok(())
    }

    /// Apply operator color depending on options.
    #[must_use]
    #[inline(always)]
    fn operator(
        text: &str,
        options: &PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        Self::fg::<Yellow>(text, options, f)
    }

    /// Apply a compile-time color depending on options.
    #[must_use]
    #[inline(always)]
    fn fg<C: Color>(
        text: &str,
        options: &PrintOptions,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if options.colors {
            write!(f, "{}", text.fg::<C>())?;
        } else {
            write!(f, "{}", text)?;
        }
        Ok(())
    }
}

/// Utility struct implementing various ascii art printing methods.
#[derive(Debug, Clone)]
pub struct PrettyPrinter {
    /// Vector of multiple lines, allowing to easily add content at the top, left, etc...
    pub lines: Vec<String>,
    width: usize,
    /// Give the height of the base of the expression.
    pub baseline: usize,
}

impl From<String> for PrettyPrinter {
    fn from(value: String) -> Self {
        PrettyPrinter {
            width: value.len(),
            lines: vec![value],
            baseline: 0,
        }
    }
}

/// Hashmap which contains unicode drawing symbols associated to characters.
static DRAWINGS_CHARS: phf::Map<char, (char, char, char)> = phf_map! {
    '(' => ('⎛', '⎜', '⎝'),
    ')' => ('⎞', '⎟', '⎠'),
    '[' => ('⎡', '⎢', '⎣'),
    ']' => ('⎤', '⎥', '⎦'),
    '|' => ('│', '│', '│')
};

impl PrettyPrinter {
    /// Create a new blank pretty printer.
    pub fn empty() -> Self {
        PrettyPrinter {
            lines: vec![],
            width: 0,
            baseline: 0,
        }
    }
    /// Check if the pretty printer is blank.
    pub fn is_empty(&self) -> bool {
        self.width == 0
    }

    /// Center horizontally and vertically the printer to be exactly of size (`height`*`width`)
    ///
    /// Panics if height or width is smaller than printer's size.
    pub fn center(&mut self, height: usize, width: usize, baseline: usize) {
        if height < self.height() || width < self.width() {
            panic!("Can't center self on a smaller size !");
        }
        if baseline > height {
            panic!("Can't have a baseline greater than the height !")
        }
        // Center vertically, aligning baseline.
        let base_offset = baseline - self.baseline;
        for _ in 0..base_offset {
            self.lines.insert(0, " ".repeat(self.width()));
        }
        let end_offset = height - self.height();
        for _ in 0..end_offset {
            self.lines.push(" ".repeat(self.width()));
        }

        // Center horizontally
        let diff = width - self.width();
        let offset = diff / 2;
        let end_offset = diff - offset;
        for line in &mut self.lines {
            line.insert_str(0, &" ".repeat(offset));
            line.push_str(&" ".repeat(end_offset));
        }
        self.width = width;
    }

    /// Concat two pretty printer, aligning their baseline and adding sym between them. `space` can be used to specify if spaces are needed around `sym`.
    ///
    /// self + other
    pub fn concat(&mut self, sym: &str, space: bool, other: &Self) {
        if self.is_empty() {
            *self = other.clone();
            return;
        }

        // Baselines should be aligned
        let other_offset = if self.baseline < other.baseline {
            for _ in 0..other.baseline - self.baseline {
                self.lines.insert(0, " ".repeat(self.width));
            }
            self.baseline = other.baseline;
            0
        } else {
            self.baseline - other.baseline
        };
        if self.height() < other.height() + other_offset {
            // Add blank lines
            for _ in 0..other.height() + other_offset - self.height() {
                self.lines.push(" ".repeat(self.width()));
            }
        }

        // Add space and the symbol
        let (sym, space) = if space {
            (format!(" {} ", sym), " ".repeat(sym.chars().count() + 2))
        } else {
            (format!("{}", sym), " ".repeat(sym.chars().count()))
        };
        for i in 0..self.height() {
            if i == self.baseline {
                self.lines[i] += &sym;
            } else {
                self.lines[i] += &space;
            }
        }

        // Concat the other printer
        for i in 0..self.height() {
            if i >= other_offset && i - other_offset < other.height() {
                self.lines[i] += &other.lines[i - other_offset];
            } else {
                self.lines[i] += &" ".repeat(other.width());
            }
        }

        // Update printer infos
        self.width += other.width + space.len();
    }

    /// Concat two pretty printer vertically, centering them.
    ///
    /// self
    /// -----
    /// other
    pub fn vertical_concat(&mut self, sym: &str, other: &Self) {
        if self.is_empty() {
            *self = other.clone();
            return;
        }

        // Center the smallest
        let diff = self.width().abs_diff(other.width());
        let offset = diff / 2;
        let offset_end = diff - offset;
        let (other_offset, other_offset_end) = if self.width() < other.width() {
            for i in &mut self.lines {
                i.insert_str(0, &" ".repeat(offset));
            }
            for i in &mut self.lines {
                *i += &" ".repeat(offset_end);
            }
            self.width += diff;
            (0, 0)
        } else {
            (offset, offset_end)
        };

        // Add the symbol
        self.baseline = self.height();
        self.lines.push(sym.repeat(self.width()));

        // Concat the other printer
        for i in 0..other.height() {
            let mut line = other.lines[i].clone();
            line.insert_str(0, &" ".repeat(other_offset));
            line += &" ".repeat(other_offset_end);
            self.lines.push(line);
        }
    }
    /// Put other as an exposant of self.
    ///
    ///     other
    /// self
    pub fn pow(&mut self, other: &Self) {
        // Add space to concat other
        self.baseline += other.height();
        for _ in 0..other.height() {
            self.lines.insert(0, " ".repeat(self.width()));
        }
        for i in 0..self.height() {
            if i < other.height() {
                self.lines[i].push_str(&other.lines[i]);
            } else {
                self.lines[i].push_str(&" ".repeat(other.width()));
            }
        }
        self.width += other.width;
    }
    /// Add `sym` at the left of self, converting it to unicode drawings chars.
    pub fn left(&mut self, sym: char) {
        let or = &(sym, sym, sym);
        let height = self.height();
        let drawings = if height == 1 {
            // If height is 1, use directly sym, not a drawing variant
            or
        } else {
            DRAWINGS_CHARS.get(&sym).unwrap_or(or)
        };
        for i in 0..height {
            self.lines[i].insert(
                0,
                if i == 0 {
                    drawings.0
                } else if i == height - 1 {
                    drawings.2
                } else {
                    drawings.1
                },
            );
        }
        self.width += 1;
    }
    /// Add `sym` at the left of self, converting it to unicode drawings chars.
    pub fn right(&mut self, sym: char) {
        let or = &(sym, sym, sym);
        let height = self.height();
        let drawings = if height == 1 {
            // If height is 1, use directly sym, not a drawing variant
            or
        } else {
            DRAWINGS_CHARS.get(&sym).unwrap_or(or)
        };
        for i in 0..height {
            self.lines[i].push(if i == 0 {
                drawings.0
            } else if i == height - 1 {
                drawings.2
            } else {
                drawings.1
            });
        }
        self.width += 1;
    }
    /// Wrap self between `start` and `end` symbol.
    pub fn group(&mut self, start: char, end: char) {
        self.left(start);
        self.right(end);
    }
    /// Wrap self between paren.
    pub fn paren(&mut self) {
        self.group('(', ')');
    }
    /// Return the width of the pretty printer.
    pub fn width(&self) -> usize {
        self.width
    }
    /// Return the height of the pretty printer.
    pub fn height(&self) -> usize {
        self.lines.len()
    }
}

impl Display for PrettyPrinter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, line) in self.lines.iter().enumerate() {
            write!(f, "{}", line)?;
            if i != self.height() - 1 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
