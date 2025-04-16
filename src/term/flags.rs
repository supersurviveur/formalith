/// Flags marking a [Term] as already normalized
pub(crate) const NORMALIZED: u8 = 0x1;

/// Util trait to manage expression's flags
pub(crate) trait Flags {
    /// Return the flags of the expression
    fn get_flags(&self) -> u8;
    /// Return a mutable reference to the flags of the expression
    fn get_flags_mut(&mut self) -> &mut u8;
    /// Add a flag to the expression
    fn add_flag(&mut self, flag: u8) {
        *self.get_flags_mut() |= flag;
    }
    /// Remove a flag from the expression
    fn remove_flag(&mut self, flag: u8) {
        *self.get_flags_mut() &= !flag;
    }
    /// If `normalized` is `true`, set the expression as already normalized (see [Term::normalize])
    fn set_normalized(&mut self, normalized: bool) {
        if normalized {
            self.add_flag(NORMALIZED);
        } else {
            self.remove_flag(NORMALIZED);
        }
    }
    /// Return `true` if the expression needs to be normalized
    fn needs_normalization(&self) -> bool {
        (self.get_flags() & NORMALIZED) != NORMALIZED
    }
}
