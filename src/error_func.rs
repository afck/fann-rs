use fann_sys::*;

/// Error function used during training.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ErrorFunc {
    /// Standard linear error function
    Linear,
    /// Tanh error function; usually better but may require a lower learning rate. This error
    /// function aggressively targets outputs that differ much from the desired, while not targeting
    /// outputs that only differ slightly. Not recommended for cascade or incremental training.
    Tanh,
}

impl ErrorFunc {
    /// Create an `ErrorFunc` from a `fann_sys::fann_errorfunc_enum`.
    pub fn from_errorfunc_enum(ef_enum: fann_errorfunc_enum) -> ErrorFunc {
        match ef_enum {
            FANN_ERRORFUNC_LINEAR => ErrorFunc::Linear,
            FANN_ERRORFUNC_TANH   => ErrorFunc::Tanh,
        }
    }

    /// Return the `fann_sys::fann_errorfunc_enum` corresponding to this `ErrorFunc`.
    pub fn to_errorfunc_enum(&self) -> fann_errorfunc_enum {
        match *self {
            ErrorFunc::Linear => FANN_ERRORFUNC_LINEAR,
            ErrorFunc::Tanh   => FANN_ERRORFUNC_TANH,
        }
    }
}
