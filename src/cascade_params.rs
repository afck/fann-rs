use activation_func::ActivationFunc;
use fann_sys::fann_type;
use libc::{c_float, c_uint};

/// Parameters for cascade training.
#[derive(Clone, Debug, PartialEq)]
pub struct CascadeParams {
    /// A number between 0 and 1 determining how large a fraction the mean square error should
    /// change within `output_stagnation_epochs` during training of the output connections, in
    /// order for the training to stagnate. After stagnation, training of the output connections
    /// ends and new candidates are prepared.
    ///
    /// This means: If the MSE does not change by a fraction of `output_change_fraction` during a
    /// period of `output_stagnation_epochs`, the training of the output connections is stopped
    /// because training has stagnated.
    pub output_change_fraction: c_float,
    /// The number of epochs training is allowed to continue without changing the MSE by a fraction
    /// of at least `output_change_fraction`.
    pub output_stagnation_epochs: c_uint,
    /// A number between 0 and 1 determining how large a fraction the mean square error should
    /// change within `candidate_stagnation_epochs` during training of the candidate neurons, in
    /// order for the training to stagnate. After stagnation, training of the candidate neurons is
    /// stopped and the best candidate is selected.
    ///
    /// This means: If the MSE does not change by a fraction of `candidate_change_fraction` during
    /// a period of `candidate_stagnation_epochs`, the training of the candidate neurons is stopped
    /// because training has stagnated.
    pub candidate_change_fraction: c_float,
    /// The number of epochs training is allowed to continue without changing the MSE by a fraction
    /// of `candidate_change_fraction`.
    pub candidate_stagnation_epochs: c_uint,
    /// A limit for how much the candidate neuron may be trained. It limits the ratio between the
    /// MSE and the candidate score.
    pub candidate_limit: fann_type,
    /// Multiplier for the weight of the candidate neuron before adding it to the network. Usually
    /// between 0 and 1, to make training less aggressive.
    pub weight_multiplier: fann_type,
    /// The maximum number of epochs the output connections may be trained after adding a new
    /// candidate neuron.
    pub max_out_epochs: c_uint,
    /// The maximum number of epochs the input connections to the candidates may be trained before
    /// adding a new candidate neuron.
    pub max_cand_epochs: c_uint,
    /// The activation functions for the candidate neurons.
    pub activation_functions: Vec<ActivationFunc>,
    /// The activation function steepness values for the candidate neurons.
    pub activation_steepnesses: Vec<fann_type>,
    /// The number of candidate neurons to be trained for each combination of activation function
    /// and steepness.
    pub num_candidate_groups: c_uint,
}

impl CascadeParams {
    /// The number of candidates used during training: the number of combinations of activation
    /// functions and steepnesses, times `num_candidate_groups`.
    ///
    /// For every combination of activation function and steepness, `num_candidate_groups` such
    /// neurons, with different initial weights, are trained.
    pub fn get_num_candidates(&self) -> c_uint {
        self.activation_functions.len() as c_uint
            * self.activation_steepnesses.len() as c_uint
            * self.num_candidate_groups
    }
}

impl Default for CascadeParams {
    fn default() -> CascadeParams {
        CascadeParams {
            output_change_fraction: 0.01,
            output_stagnation_epochs: 12,
            candidate_change_fraction: 0.01,
            candidate_stagnation_epochs: 12,
            candidate_limit: 1000.0,
            weight_multiplier: 0.4,
            max_out_epochs: 150,
            max_cand_epochs: 150,
            activation_functions: vec!(ActivationFunc::Sigmoid,
                                       ActivationFunc::SigmoidSymmetric,
                                       ActivationFunc::Gaussian,
                                       ActivationFunc::GaussianSymmetric,
                                       ActivationFunc::Elliott,
                                       ActivationFunc::ElliottSymmetric,
                                       ActivationFunc::SinSymmetric,
                                       ActivationFunc::CosSymmetric,
                                       ActivationFunc::Sin,
                                       ActivationFunc::Cos),
            activation_steepnesses: vec!(0.25, 0.5, 0.75, 1.0),
            num_candidate_groups: 2,
        }
    }
}

