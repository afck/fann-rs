use fann_sys::*;
pub use error::{FannError, FannErrorType, FannResult};

/// The activation functions used for the neurons during training. They can either be set for a
/// group of neurons using `set_activation_func_hidden` and `set_activation_func_output`, or for a
/// single neuron using `set_activation_func`.
///
/// Similarly, the steepness of an activation function is specified using
/// `set_activation_steepness_hidden`, `set_activation_steepness_output` and
/// `set_activation_steepness`.
///
/// In the descriptions of the functions:
///
/// * x is the input to the activation function,
///
/// * y is the output,
///
/// * s is the steepness and
///
/// * d is the derivation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ActivationFunc {
    /// Linear activation function.
    ///
    /// * span: -inf < y < inf
    ///
    /// * y = x*s, d = 1*s
    ///
    /// * Can NOT be used in fixed point.
    Linear,
    /// Threshold activation function.
    ///
    /// * x < 0 -> y = 0, x >= 0 -> y = 1
    ///
    /// * Can NOT be used during training.
    Threshold,
    /// Threshold activation function.
    ///
    /// * x < 0 -> y = 0, x >= 0 -> y = 1
    ///
    /// * Can NOT be used during training.
    ThresholdSymmetric,
    /// Sigmoid activation function.
    ///
    /// * One of the most used activation functions.
    ///
    /// * span: 0 < y < 1
    ///
    /// * y = 1/(1 + exp(-2*s*x))
    ///
    /// * d = 2*s*y*(1 - y)
    Sigmoid,
    /// Stepwise linear approximation to sigmoid.
    ///
    /// * Faster than sigmoid but a bit less precise.
    SigmoidStepwise,
    /// Symmetric sigmoid activation function, aka. tanh.
    ///
    /// * One of the most used activation functions.
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
    ///
    /// * d = s*(1-(y*y))
    SigmoidSymmetric,
    /// Stepwise linear approximation to symmetric sigmoid.
    ///
    /// * Faster than symmetric sigmoid but a bit less precise.
    SigmoidSymmetricStepwise,
    /// Gaussian activation function.
    ///
    /// * 0 when x = -inf, 1 when x = 0 and 0 when x = inf
    ///
    /// * span: 0 < y < 1
    ///
    /// * y = exp(-x*s*x*s)
    ///
    /// * d = -2*x*s*y*s
    Gaussian,
    /// Symmetric gaussian activation function.
    ///
    /// * -1 when x = -inf, 1 when x = 0 and 0 when x = inf
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = exp(-x*s*x*s)*2-1
    ///
    /// * d = -2*x*s*(y+1)*s
    GaussianSymmetric,
    /// Stepwise linear approximation to gaussian.
    /// Faster than gaussian but a bit less precise.
    /// NOT implemented yet.
    GaussianStepwise,
    /// Fast (sigmoid like) activation function defined by David Elliott
    ///
    /// * span: 0 < y < 1
    ///
    /// * y = ((x*s) / 2) / (1 + |x*s|) + 0.5
    ///
    /// * d = s*1/(2*(1+|x*s|)*(1+|x*s|))
    Elliott,
    /// Fast (symmetric sigmoid like) activation function defined by David Elliott
    ///
    /// * span: -1 < y < 1
    ///
    /// * y = (x*s) / (1 + |x*s|)
    ///
    /// * d = s*1/((1+|x*s|)*(1+|x*s|))
    ElliottSymmetric,
    /// Bounded linear activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = x*s, d = 1*s
    LinearPiece,
    /// Bounded linear activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = x*s, d = 1*s
    LinearPieceSymmetric,
    /// Periodical sine activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = sin(x*s)
    ///
    /// * d = s*cos(x*s)
    SinSymmetric,
    /// Periodical cosine activation function.
    ///
    /// * span: -1 <= y <= 1
    ///
    /// * y = cos(x*s)
    ///
    /// * d = s*-sin(x*s)
    CosSymmetric,
    /// Periodical sine activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = sin(x*s)/2+0.5
    ///
    /// * d = s*cos(x*s)/2
    Sin,
    /// Periodical cosine activation function.
    ///
    /// * span: 0 <= y <= 1
    ///
    /// * y = cos(x*s)/2+0.5
    ///
    /// * d = s*-sin(x*s)/2
    Cos,
}

impl ActivationFunc {
    /// Create an `ActivationFunc` from a `fann_sys::fann_activationfunc_enum`.
    pub fn from_fann_activationfunc_enum(af_enum: fann_activationfunc_enum)
            -> FannResult<ActivationFunc> {
        match af_enum {
            FANN_NONE => Err(FannError {
                             error_type: FannErrorType::IndexOutOfBound,
                             error_str: "Neuron or layer index is out of bound.".to_string(),
                         }),
            FANN_LINEAR                     => Ok(ActivationFunc::Linear),
            FANN_THRESHOLD                  => Ok(ActivationFunc::Threshold),
            FANN_THRESHOLD_SYMMETRIC        => Ok(ActivationFunc::ThresholdSymmetric),
            FANN_SIGMOID                    => Ok(ActivationFunc::Sigmoid),
            FANN_SIGMOID_STEPWISE           => Ok(ActivationFunc::SigmoidStepwise),
            FANN_SIGMOID_SYMMETRIC          => Ok(ActivationFunc::SigmoidSymmetric),
            FANN_SIGMOID_SYMMETRIC_STEPWISE => Ok(ActivationFunc::SigmoidSymmetricStepwise),
            FANN_GAUSSIAN                   => Ok(ActivationFunc::Gaussian),
            FANN_GAUSSIAN_SYMMETRIC         => Ok(ActivationFunc::GaussianSymmetric),
            FANN_GAUSSIAN_STEPWISE          => Ok(ActivationFunc::GaussianStepwise),
            FANN_ELLIOTT                    => Ok(ActivationFunc::Elliott),
            FANN_ELLIOTT_SYMMETRIC          => Ok(ActivationFunc::ElliottSymmetric),
            FANN_LINEAR_PIECE               => Ok(ActivationFunc::LinearPiece),
            FANN_LINEAR_PIECE_SYMMETRIC     => Ok(ActivationFunc::LinearPieceSymmetric),
            FANN_SIN_SYMMETRIC              => Ok(ActivationFunc::SinSymmetric),
            FANN_COS_SYMMETRIC              => Ok(ActivationFunc::CosSymmetric),
            FANN_SIN                        => Ok(ActivationFunc::Sin),
            FANN_COS                        => Ok(ActivationFunc::Cos),
        }
    }

    /// Return the `fann_sys::fann_activationfunc_enum` corresponding to this `ActivationFunc`.
    pub fn to_fann_activationfunc_enum(&self) -> fann_activationfunc_enum {
        match *self {
            ActivationFunc::Linear                   => FANN_LINEAR,
            ActivationFunc::Threshold                => FANN_THRESHOLD,
            ActivationFunc::ThresholdSymmetric       => FANN_THRESHOLD_SYMMETRIC,
            ActivationFunc::Sigmoid                  => FANN_SIGMOID,
            ActivationFunc::SigmoidStepwise          => FANN_SIGMOID_STEPWISE,
            ActivationFunc::SigmoidSymmetric         => FANN_SIGMOID_SYMMETRIC,
            ActivationFunc::SigmoidSymmetricStepwise => FANN_SIGMOID_SYMMETRIC_STEPWISE,
            ActivationFunc::Gaussian                 => FANN_GAUSSIAN,
            ActivationFunc::GaussianSymmetric        => FANN_GAUSSIAN_SYMMETRIC,
            ActivationFunc::GaussianStepwise         => FANN_GAUSSIAN_STEPWISE,
            ActivationFunc::Elliott                  => FANN_ELLIOTT,
            ActivationFunc::ElliottSymmetric         => FANN_ELLIOTT_SYMMETRIC,
            ActivationFunc::LinearPiece              => FANN_LINEAR_PIECE,
            ActivationFunc::LinearPieceSymmetric     => FANN_LINEAR_PIECE_SYMMETRIC,
            ActivationFunc::SinSymmetric             => FANN_SIN_SYMMETRIC,
            ActivationFunc::CosSymmetric             => FANN_COS_SYMMETRIC,
            ActivationFunc::Sin                      => FANN_SIN,
            ActivationFunc::Cos                      => FANN_COS,
        }
    }
}

