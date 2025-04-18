//! Contexts and symbols implementation.
//! Context are used to store a hashmap linking a variable name to its unique identifier. A same variable can have multiple names.

use std::{
    collections::HashMap,
    sync::{LazyLock, RwLock},
};

use append_only_vec::AppendOnlyVec;

/// Store all symbols' data created
static SYMBOLS: AppendOnlyVec<SymbolData> = AppendOnlyVec::new();

/// Store the global context containing symbols' data
static GLOBAL_CONTEXT: LazyLock<RwLock<Context>> = LazyLock::new(|| RwLock::new(Context::new()));

/// A symbol, representing a named variable.
///
/// See [crate::symbol!] to create a new symbol.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Symbol(usize);

impl Symbol {
    /// Create a symbol by name in the global context.
    ///
    /// If the symbol already exists, his identifier is returned, otherwise a new symbol is added in the global context.
    /// ```
    /// use formalith::context::Symbol;
    ///
    /// let x = Symbol::new("x".into());
    /// let x2 = Symbol::new("x".into());
    /// let y = Symbol::new("y".into());
    /// let another_symbol = Symbol::new("aReallyLongSymbol".into());
    ///
    /// assert_eq!(x, x2);
    /// assert_ne!(x, y);
    /// ```
    pub fn new(name: String) -> Self {
        Context::get_global_context()
            .write()
            .unwrap()
            .get_symbol(name)
    }
}

// Create a symbol by name in the global context.
/// If the symbol already exists, his identifier is returned, otherwise a new symbol is added in the global context.
/// See [Symbol]
/// ```
/// use formalith::symbol;
///
/// let x = symbol!("x");
/// let x2 = symbol!("x");
/// let y = symbol!("y");
/// let another_symbol = symbol!("aReallyLongSymbol");
///
/// assert_eq!(x, x2);
/// assert_ne!(x, y);
/// ```
#[macro_export(local_inner_macros)]
macro_rules! symbol {
    ( $x:expr ) => {
        $crate::context::Symbol::new($x.into())
    };
    ( $x:expr, $f:expr) => {
        $crate::term::Term::Symbol($crate::term::SymbolTerm::new(
            $crate::context::Symbol::new($x.into()),
            $f,
        ))
    };
}

/// Stores data associated with a symbol.
#[derive(Clone, Debug)]
pub struct SymbolData {
    /// The symbol visual representation
    pub name: String,
}

/// A context storing variables names in this specific context. By default
pub struct Context {
    symbols: HashMap<String, Symbol>,
}

impl Context {
    /// Absolute function
    pub const ABS: Symbol = Symbol(0);

    const BUILTIN_NAMES: [&'static str; 1] = ["abs"];

    /// Create a new empty context
    pub fn new() -> Self {
        let mut ctx = Self {
            symbols: HashMap::new(),
        };
        for name in Self::BUILTIN_NAMES {
            ctx.get_symbol(name.to_string());
        }
        ctx
    }

    /// Returns a reference to the global static context, used by default
    pub(crate) fn get_global_context() -> &'static RwLock<Self> {
        &GLOBAL_CONTEXT
    }

    /// Return a [Symbol] from its name in a context
    // / ```
    // / use formalith::context::Context;
    // /
    // / // Create a new global symbol
    // / let x = symbol!("x");
    // / assert_eq!(Context::get_global_context().get_symbol("x"), x);
    // / ```
    pub fn get_symbol(&mut self, name: String) -> Symbol {
        self.symbols
            .entry(name)
            .or_insert_with_key(|name| Symbol(SYMBOLS.push(SymbolData { name: name.clone() })))
            .clone()
    }
    /// Return a reference to the [SymbolData] of a given [Symbol]
    pub fn get_symbol_data(symbol: &Symbol) -> &'static SymbolData {
        &SYMBOLS[symbol.0]
    }
}
