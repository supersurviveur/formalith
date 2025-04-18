//! Printing trait and options to render expressions.

use std::fmt::Display;

use owo_colors::{
    colors::{BrightBlack, Yellow},
    Color, OwoColorize,
};

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
            pretty_print: false,
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
pub struct PrettyPrinter {
    /// Vector of multiple lines, allowing to easily add content at the top, left, etc...
    pub lines: Vec<String>,
    pub width: usize,
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

impl PrettyPrinter {
    pub fn concat(&mut self, sym: &str, other: &Self) {
        // Baselines should be aligned
        let other_offset = if self.baseline < other.baseline {
            for _ in 0..other.baseline - self.baseline {
                self.lines.insert(0, " ".repeat(self.width));
            }
            0
        } else {
            self.baseline - other.baseline
        };

        // Add space and the symbol
        for i in 0..self.height() {
            if i == self.baseline {
                self.lines[i] += &format!(" {} ", sym);
            } else {
                self.lines[i] += "   ";
            }
        }

        // Concat the other printer
        for i in 0..self.height() {
            if i >= other_offset {
                self.lines[i] += &other.lines[i - other_offset];
            }
        }

        // Update printer infos
        self.width += other.width + 3;
    }

    pub fn height(&self) -> usize {
        self.lines.len()
    }
}

impl Display for PrettyPrinter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for line in &self.lines {
            write!(f, "{}", line)?;
        }
        Ok(())
    }
}
