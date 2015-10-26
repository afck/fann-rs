use fann_sys::*;

/// Stop critieria for training.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StopFunc {
    /// The mean square error of the whole output.
    Mse,
    /// The number of training data points where the output neuron's error was greater than the bit
    /// fail limit. Every neuron is counted for every training data sample where it fails.
    Bit,
}

impl StopFunc {
    /// Create a `StopFunc` from a `fann_sys::fann_stopfunc_enum`.
    pub fn from_stopfunc_enum(sf_enum: fann_stopfunc_enum) -> StopFunc {
        match sf_enum {
            FANN_STOPFUNC_MSE => StopFunc::Mse,
            FANN_STOPFUNC_BIT => StopFunc::Bit,
        }
    }

    /// Return the `fann_sys::fann_stopfunc_enum` corresponding to this `StopFunc`.
    pub fn to_stopfunc_enum(&self) -> fann_stopfunc_enum {
        match *self {
            StopFunc::Mse => FANN_STOPFUNC_MSE,
            StopFunc::Bit => FANN_STOPFUNC_BIT,
        }
    }
}
