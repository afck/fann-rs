extern crate libc;
extern crate fann_sys;

use error::{FannError, FannErrorType};
use fann_sys::fann_activationfunc_enum::*;
use fann_sys::fann_errorfunc_enum::*;
use fann_sys::fann_nettype_enum::*;
use fann_sys::fann_stopfunc_enum::*;
use fann_sys::fann_train_enum::*;
use fann_sys::fann_type;
use libc::{c_float, c_int, c_uint};
use std::path::Path;
use std::ptr::copy_nonoverlapping;
use train_data::*;

pub mod error;
pub mod train_data;

/// The Training algorithms used when training on `fann_train_data` with functions like
/// `fann_train_on_data` or `fann_train_on_file`. The incremental training alters the weights
/// after each time it is presented an input pattern, while batch only alters the weights once after
/// it has been presented to all the patterns.
#[derive(Copy, Clone)]
pub enum TrainAlgorithm {
    /// Standard backpropagation algorithm, where the weights are updated after each training
    /// pattern. This means that the weights are updated many times during a single epoch and some
    /// problems will train very fast, while other more advanced problems will not train very well.
    Incremental,
    /// Standard backpropagation algorithm, where the weights are updated after calculating the mean
    /// square error for the whole training set. This means that the weights are only updated once
    /// during an epoch. For this reason some problems will train slower with this algorithm. But
    /// since the mean square error is calculated more correctly than in incremental training, some
    /// problems will reach better solutions.
    Batch,
    /// A more advanced batch training algorithm which achieves good results for many problems.
    /// `Rprop` is adaptive and therefore does not use the `learning_rate`. Some other parameters
    /// can, however, be set to change the way `Rprop` works, but it is only recommended for users
    /// with a deep understanding of the algorithm. The original RPROP training algorithm is
    /// described by [Riedmiller and Braun, 1993], but the algorithm used here is a variant, iRPROP,
    /// described by [Igel and Husken, 2000].
    Rprop,
    /// A more advanced batch training algorithm which achieves good results for many problems. The
    /// quickprop training algorithm uses the `learning_rate` parameter along with other more
    /// advanced parameters, but it is only recommended to change these for users with a deep
    /// understanding of the algorithm. Quickprop is described by [Fahlman, 1988].
    Quickprop,
}

impl TrainAlgorithm {
    fn from_fann_train_enum(fte: fann_sys::fann_train_enum) -> TrainAlgorithm {
        match fte {
            FANN_TRAIN_INCREMENTAL => TrainAlgorithm::Incremental,
            FANN_TRAIN_BATCH       => TrainAlgorithm::Batch,
            FANN_TRAIN_RPROP       => TrainAlgorithm::Rprop,
            FANN_TRAIN_QUICKPROP   => TrainAlgorithm::Quickprop,
        }
    }

    fn to_fann_train_enum(&self) -> fann_sys::fann_train_enum {
        match *self {
            TrainAlgorithm::Incremental => FANN_TRAIN_INCREMENTAL,
            TrainAlgorithm::Batch       => FANN_TRAIN_BATCH,
            TrainAlgorithm::Rprop       => FANN_TRAIN_RPROP,
            TrainAlgorithm::Quickprop   => FANN_TRAIN_QUICKPROP,
        }
    }
}

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
#[derive(Copy, Clone)]
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
    fn from_fann_activationfunc_enum(af_enum: fann_sys::fann_activationfunc_enum)
            -> Option<ActivationFunc> {
        match af_enum {
            // TODO: FANN_NONE                       => None,
            FANN_LINEAR                     => Some(ActivationFunc::Linear),
            FANN_THRESHOLD                  => Some(ActivationFunc::Threshold),
            FANN_THRESHOLD_SYMMETRIC        => Some(ActivationFunc::ThresholdSymmetric),
            FANN_SIGMOID                    => Some(ActivationFunc::Sigmoid),
            FANN_SIGMOID_STEPWISE           => Some(ActivationFunc::SigmoidStepwise),
            FANN_SIGMOID_SYMMETRIC          => Some(ActivationFunc::SigmoidSymmetric),
            FANN_SIGMOID_SYMMETRIC_STEPWISE => Some(ActivationFunc::SigmoidSymmetricStepwise),
            FANN_GAUSSIAN                   => Some(ActivationFunc::Gaussian),
            FANN_GAUSSIAN_SYMMETRIC         => Some(ActivationFunc::GaussianSymmetric),
            FANN_GAUSSIAN_STEPWISE          => Some(ActivationFunc::GaussianStepwise),
            FANN_ELLIOTT                    => Some(ActivationFunc::Elliott),
            FANN_ELLIOTT_SYMMETRIC          => Some(ActivationFunc::ElliottSymmetric),
            FANN_LINEAR_PIECE               => Some(ActivationFunc::LinearPiece),
            FANN_LINEAR_PIECE_SYMMETRIC     => Some(ActivationFunc::LinearPieceSymmetric),
            FANN_SIN_SYMMETRIC              => Some(ActivationFunc::SinSymmetric),
            FANN_COS_SYMMETRIC              => Some(ActivationFunc::CosSymmetric),
            FANN_SIN                        => Some(ActivationFunc::Sin),
            FANN_COS                        => Some(ActivationFunc::Cos),
        }
    }

    fn to_fann_activationfunc_enum(&self) -> fann_sys::fann_activationfunc_enum {
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

/// Error function used during training.
#[derive(Copy, Clone)]
pub enum ErrorFunc {
    /// Standard linear error function
    Linear,
    /// Tanh error function; usually better but may require a lower learning rate. This error
    /// function aggressively targets outputs that differ much from the desired, while not targeting
    /// outputs that only differ slightly. Not recommended for cascade or incremental training.
    Tanh,
}

impl ErrorFunc {
    fn from_errorfunc_enum(ef_enum: fann_sys::fann_errorfunc_enum) -> ErrorFunc {
        match ef_enum {
            FANN_ERRORFUNC_LINEAR => ErrorFunc::Linear,
            FANN_ERRORFUNC_TANH   => ErrorFunc::Tanh,
        }
    }

    fn to_errorfunc_enum(&self) -> fann_sys::fann_errorfunc_enum {
        match *self {
            ErrorFunc::Linear => FANN_ERRORFUNC_LINEAR,
            ErrorFunc::Tanh   => FANN_ERRORFUNC_TANH,
        }
    }
}

/// Stop critieria for training.
#[derive(Copy, Clone)]
pub enum StopFunc {
    /// The mean square error of the whole output.
    Mse,
    /// The number of training data points where the output neuron's error was greater than the bit
    /// fail limit. Every neuron is counted for every training data sample where it fails.
    Bit,
}

impl StopFunc {
    fn from_stopfunc_enum(sf_enum: fann_sys::fann_stopfunc_enum) -> StopFunc {
        match sf_enum {
            FANN_STOPFUNC_MSE => StopFunc::Mse,
            FANN_STOPFUNC_BIT => StopFunc::Bit,
        }
    }

    fn to_stopfunc_enum(&self) -> fann_sys::fann_stopfunc_enum {
        match *self {
            StopFunc::Mse => FANN_STOPFUNC_MSE,
            StopFunc::Bit => FANN_STOPFUNC_BIT,
        }
    }
}

/// Network types
#[derive(Copy, Clone)]
pub enum NetType {
    /// Each layer of neurons only has connections to the next layer.
    Layer,
    /// Each layer has connections to all following layers.
    Shortcut,
}

impl NetType {
    fn from_nettype_enum(nt_enum: fann_sys::fann_nettype_enum) -> NetType {
        match nt_enum {
            FANN_NETTYPE_LAYER    => NetType::Layer,
            FANN_NETTYPE_SHORTCUT => NetType::Shortcut,
        }
    }
}

pub struct Fann {
    // We don't consider setting and clearing the error string and number a mutation, and every
    // method should leave these fields cleared, either because it succeeded or because it read the
    // fields and returned the corresponding error.
    // We also don't consider writing the output data a mutation, as we don't provide access to it
    // and copy it before returning it.
    raw: *mut fann_sys::fann,
}

impl Fann {
    /// Create a fully connected neural network.
    ///
    /// There will be a bias neuron in each layer except the output layer,
    /// and this bias neuron will be connected to all neurons in the next layer.
    /// When running the network, the bias nodes always emit 1.
    ///
    /// # Arguments
    ///
    /// * `layers` - Specifies the number of neurons in each layer, starting with the input and
    ///              ending with the output layer.
    ///
    /// # Example
    ///
    /// ```
    /// // Creating a network with 2 input neurons, 1 output neuron,
    /// // and two hidden layers with 8 and 9 neurons.
    /// let layers = [2, 8, 9, 1];
    /// fann::Fann::new(&layers).unwrap();
    /// ```
    pub fn new(layers: &[c_uint]) -> Result<Fann, FannError> {
        Fann::new_sparse(1.0, layers)
    }

    /// Create a neural network that is not necessarily fully connected.
    ///
    /// There will be a bias neuron in each layer except the output layer,
    /// and this bias neuron will be connected to all neurons in the next layer.
    /// When running the network, the bias nodes always emit 1.
    ///
    /// # Arguments
    ///
    /// * `connection_rate` - The share of pairs of neurons in consecutive layers that will be
    ///                       connected.
    /// * `layers`          - Specifies the number of neurons in each layer, starting with the input
    ///                       and ending with the output layer.
    pub fn new_sparse(connection_rate: c_float, layers: &[c_uint]) -> Result<Fann, FannError> {
        unsafe {
            let raw = fann_sys::fann_create_sparse_array(connection_rate,
                                                         layers.len() as c_uint,
                                                         layers.as_ptr());
            try!(FannError::check_no_error(raw as *mut fann_sys::fann_error));
            Ok(Fann { raw: raw })
        }
    }

    /// Create a neural network which has shortcut connections, i. e. it doesn't connect only each
    /// layer to its successor, but every layer with every later layer: Each neuron has connections
    /// to all neurons in all subsequent layers.
    pub fn new_shortcut(layers: &[c_uint]) -> Result<Fann, FannError> {
        unsafe {
            let raw = fann_sys::fann_create_shortcut_array(layers.len() as c_uint, layers.as_ptr());
            try!(FannError::check_no_error(raw as *mut fann_sys::fann_error));
            Ok(Fann { raw: raw })
        }
    }

    // TODO: save methods
    // TODO: from_file
    // TODO: randomize_weights
    // TODO: init_weights
    // TODO: print methods

    /// Return an `Err` if the size of the slice does not match the number of input neurons,
    /// otherwise `Ok(())`.
    fn check_input_size(&self, input: &[fann_type]) -> Result<(), FannError> {
        let num_input = self.get_num_input() as usize;
        if input.len() == num_input {
            Ok(())
        } else {
            Err(FannError {
                error_type: FannErrorType::IndexOutOfBound, // TODO: New error type?
                error_str: format!("Input has length {}, but there are {} input neurons",
                                   input.len(), num_input),
            })
        }
    }

    /// Return an `Err` if the size of the slice does not match the number of output neurons,
    /// otherwise `Ok(())`.
    fn check_output_size(&self, output: &[fann_type]) -> Result<(), FannError> {
        let num_output = self.get_num_output() as usize;
        if output.len() == num_output {
            Ok(())
        } else {
            Err(FannError {
                error_type: FannErrorType::IndexOutOfBound, // TODO: New error type?
                error_str: format!("Output has length {}, but there are {} output neurons",
                                   output.len(), num_output),
            })
        }
    }

    /// Train with a single pair of input and output. This is always incremental training (see
    /// `TrainAlg`), since only one pattern is presented.
    pub fn train(&mut self, input: &[fann_type], desired_output: &[fann_type])
            -> Result<(), FannError> {
        unsafe {
            try!(self.check_input_size(input));
            try!(self.check_output_size(desired_output));
            fann_sys::fann_train(self.raw, input.as_ptr(), desired_output.as_ptr());
            try!(FannError::check_no_error(self.raw as *mut fann_sys::fann_error));
        }
        Ok(())
    }

    /// Train the network on the given data set.
    ///
    /// # Arguments
    ///
    /// * `data`                   - The training data.
    /// * `max_epochs`             - The maximum number of training epochs.
    /// * `epochs_between_reports` - The number of epochs between printing a status report to
    ///                              `stdout`, or `0` to print no reports.
    /// * `desired_error`          - The desired maximum value of `get_mse` or `get_bit_fail`,
    ///                              depending on which stop function was selected.
    pub fn train_on_data(&mut self,
                         data: &TrainData,
                         max_epochs: c_uint,
                         epochs_between_reports: c_uint,
                         desired_error: c_float) -> Result<(), FannError> {
        unsafe {
            fann_sys::fann_train_on_data(self.raw,
                                         data.get_raw(),
                                         max_epochs,
                                         epochs_between_reports,
                                         desired_error);
            FannError::check_no_error(self.raw as *mut fann_sys::fann_error)
        }
    }

    /// Do the same as `train_on_data` but read the training data directly from a file.
    pub fn train_on_file<P: AsRef<Path>>(&mut self,
                                         path: P,
                                         max_epochs: c_uint,
                                         epochs_between_reports: c_uint,
                                         desired_error: c_float) -> Result<(), FannError> {
        let train = try!(TrainData::from_file(path));
        self.train_on_data(&train, max_epochs, epochs_between_reports, desired_error)
    }

    /// Train one epoch with a set of training data, i. e. each sample from the training data is
    /// considered exactly once.
    ///
    /// Returns the mean square error as it is calculated either before or during the actual
    /// training. This is not the actual MSE after the training epoch, but since calculating this
    /// will require to go through the entire training set once more, it is more than adequate to
    /// use this value during training.
    pub fn train_epoch(&mut self, data: &TrainData) -> Result<c_float, FannError> {
        unsafe {
            let mse = fann_sys::fann_train_epoch(self.raw, data.get_raw());
            try!(FannError::check_no_error(self.raw as *mut fann_sys::fann_error));
            Ok(mse)
        }
    }

    /// Test with a single pair of input and output. This operation updates the mean square error
    /// but does not change the network.
    ///
    /// Returns the actual output of the network.
    pub fn test(&mut self, input: &[fann_type], desired_output: &[fann_type])
            -> Result<Vec<fann_type>, FannError> {
        try!(self.check_input_size(input));
        try!(self.check_output_size(desired_output));
        let num_output = self.get_num_output() as usize;
        let mut result = Vec::with_capacity(num_output);
        unsafe {
            let output = fann_sys::fann_test(self.raw, input.as_ptr(), desired_output.as_ptr());
            try!(FannError::check_no_error(self.raw as *mut fann_sys::fann_error));
            copy_nonoverlapping(output, result.as_mut_ptr(), num_output);
            result.set_len(num_output);
        }
        Ok(result)
    }

    /// Test with a training data set and calculate the mean square error.
    pub fn test_data(&mut self, data: &TrainData) -> Result<c_float, FannError> {
        unsafe {
            let mse = fann_sys::fann_test_data(self.raw, data.get_raw());
            try!(FannError::check_no_error(self.raw as *mut fann_sys::fann_error));
            Ok(mse)
        }
    }

    /// Get the mean square error.
    pub fn get_mse(&self) -> c_float {
        unsafe { fann_sys::fann_get_MSE(self.raw) }
    }

    /// Get the number of fail bits, i. e. the number of neurons which differed from the desired
    /// output by more than the bit fail limit since the previous reset.
    pub fn get_bit_fail(&self) -> c_uint {
        unsafe { fann_sys::fann_get_bit_fail(self.raw) }
    }

    /// Reset the mean square error and bit fail count.
    pub fn reset_mse_and_bit_fail(&mut self) {
        unsafe { fann_sys::fann_reset_MSE(self.raw); }
    }

    /// Run the input through the neural network and returns the output. The length of the input
    /// must equal the number of input neurons and the length of the output will equal the number
    /// of output neurons.
    pub fn run(&self, input: &[fann_type]) -> Result<Vec<fann_type>, FannError> {
        try!(self.check_input_size(input));
        let num_output = self.get_num_output() as usize;
        let mut result = Vec::with_capacity(num_output);
        unsafe {
            let output = fann_sys::fann_run(self.raw, input.as_ptr());
            try!(FannError::check_no_error(self.raw as *mut fann_sys::fann_error));
            copy_nonoverlapping(output, result.as_mut_ptr(), num_output);
            result.set_len(num_output);
        }
        Ok(result)
    }

    /// Get the number of input neurons.
    pub fn get_num_input(&self) -> c_uint {
        unsafe { fann_sys::fann_get_num_input(self.raw) }
    }

    /// Get the number of output neurons.
    pub fn get_num_output(&self) -> c_uint {
        unsafe { fann_sys::fann_get_num_output(self.raw) }
    }

    /// Get the total number of neurons, including the bias neurons.
    ///
    /// E. g. a 2-4-2 network has 3 + 5 + 2 = 10 neurons (because two layers have bias neurons).
    pub fn get_total_neurons(&self) -> c_uint {
        unsafe { fann_sys::fann_get_total_neurons(self.raw) }
    }

    /// Get the total number of connections.
    pub fn get_total_connections(&self) -> c_uint {
        unsafe { fann_sys::fann_get_total_connections(self.raw) }
    }

    /// Get the type of the neural network.
    pub fn get_network_type(&self) -> NetType {
        let nt_enum = unsafe { fann_sys::fann_get_network_type(self.raw) };
        NetType::from_nettype_enum(nt_enum)
    }

    /// Get the connection rate used when the network was created.
    pub fn get_connection_rate(&self) -> c_float {
        unsafe { fann_sys::fann_get_connection_rate(self.raw) }
    }

    /// Get the number of layers in the network.
    pub fn get_num_layers(&self) -> c_uint {
        unsafe { fann_sys::fann_get_num_layers(self.raw) }
    }

    // TODO: get_layer_array (layer_vec?)
    // TODO: get_bias_array (bias_vec?)
    // TODO: get_connection_array (connection_vec?)
    // TODO: set_weight methods
    // TODO: user_data methods?

    /// Get the activation function for neuron number `neuron` in layer number `layer`, counting
    /// the input layer as number 0. Input layer neurons do not have an activation function, so
    /// `layer` must be at least 1.
    pub fn get_activation_func(&self, layer: c_int, neuron: c_int) -> Option<ActivationFunc> {
        let af_enum = unsafe { fann_sys::fann_get_activation_function(self.raw, layer, neuron) };
        ActivationFunc::from_fann_activationfunc_enum(af_enum)
    }

    /// Set the activation function for neuron number `neuron` in layer number `layer`, counting
    /// the input layer as number 0. Input layer neurons do not have an activation function, so
    /// `layer` must be at least 1.
    pub fn set_activation_func(&mut self, af: ActivationFunc, layer: c_int, neuron: c_int) {
        let af_enum = af.to_fann_activationfunc_enum();
        unsafe { fann_sys::fann_set_activation_function(self.raw, af_enum, layer, neuron) }
    }

    /// Set the activation function for all hidden layers.
    pub fn set_activation_func_hidden(&mut self, activation_func: ActivationFunc) {
        unsafe {
            let af_enum = activation_func.to_fann_activationfunc_enum();
            fann_sys::fann_set_activation_function_hidden(self.raw, af_enum);
        }
    }

    /// Set the activation function for the output layer.
    pub fn set_activation_func_output(&mut self, activation_func: ActivationFunc) {
        unsafe {
            let af_enum = activation_func.to_fann_activationfunc_enum();
            fann_sys::fann_set_activation_function_output(self.raw, af_enum)
        }
    }

    /// Get the activation steepness for neuron number `neuron` in layer number `layer`.
    pub fn get_activation_steepness(&self, layer: c_int, neuron: c_int) -> Option<fann_type> {
        let steepness = unsafe { fann_sys::fann_get_activation_steepness(self.raw, layer, neuron) };
        match steepness {
            -1.0 => None,
            s    => Some(s),
        }
    }

    /// Set the activation steepness for neuron number `neuron` in layer number `layer`, counting
    /// the input layer as number 0. Input layer neurons do not have an activation steepness, so
    /// layer must be at least 1.
    ///
    /// The steepness determines how fast the function goes from minimum to maximum. A higher value
    /// will result in more aggressive training.
    ///
    /// A steep activation function is adequate if outputs are binary, e. e. they are supposed to
    /// be either almost 0 or almost 1.
    ///
    /// The default value is 0.5.
    pub fn set_activation_steepness(&self, steepness: fann_type, layer: c_int, neuron: c_int) {
        unsafe { fann_sys::fann_set_activation_steepness(self.raw, steepness, layer, neuron) }
    }

    /// Set the activation steepness for layer number `layer`.
    pub fn set_activation_steepness_layer(&self, steepness: fann_type, layer: c_int) {
        unsafe { fann_sys::fann_set_activation_steepness_layer(self.raw, steepness, layer) }
    }

    /// Set the activation steepness for all hidden layers.
    pub fn set_activation_steepness_hidden(&self, steepness: fann_type) {
        unsafe { fann_sys::fann_set_activation_steepness_hidden(self.raw, steepness) }
    }

    /// Set the activation steepness for the output layer.
    pub fn set_activation_steepness_output(&self, steepness: fann_type) {
        unsafe { fann_sys::fann_set_activation_steepness_output(self.raw, steepness) }
    }

    /// Get the error function used during training.
    pub fn get_error_func(&self) -> ErrorFunc {
        let ef_enum = unsafe { fann_sys::fann_get_train_error_function(self.raw) };
        ErrorFunc::from_errorfunc_enum(ef_enum)
    }

    /// Set the error function used during training.
    ///
    /// The default is `Tanh`.
    pub fn set_error_func(&mut self, ef: ErrorFunc) {
        let ef_enum = ef.to_errorfunc_enum();
        unsafe { fann_sys::fann_set_train_error_function(self.raw, ef_enum) }
    }

    /// Get the stop criterion for training.
    pub fn get_stop_func(&self) -> StopFunc {
        let sf_enum = unsafe { fann_sys::fann_get_train_stop_function(self.raw) };
        StopFunc::from_stopfunc_enum(sf_enum)
    }

    /// Set the stop criterion for training.
    ///
    /// The default is `Mse`.
    pub fn set_stop_func(&mut self, sf: StopFunc) {
        let sf_enum = sf.to_stopfunc_enum();
        unsafe { fann_sys::fann_set_train_stop_function(self.raw, sf_enum) }
    }

    /// Get the bit fail limit.
    pub fn get_bit_fail_limit(&self) -> fann_type {
        unsafe { fann_sys::fann_get_bit_fail_limit(self.raw) }
    }

    /// Set the bit fail limit.
    ///
    /// Each output neuron value that differs from the desired output by more than the bit fail
    /// limit is counted as a failed bit.
    pub fn set_bit_fail_limit(&mut self, bit_fail_limit: fann_type) {
        unsafe { fann_sys::fann_set_bit_fail_limit(self.raw, bit_fail_limit) }
    }

    // TODO: quickprop methods
    // TODO: rprop methods
    // TODO: cascadetrain methods

    /// Get the currently configured training algorithm.
    pub fn get_train_algorithm(&self) -> TrainAlgorithm {
        let ft_enum = unsafe { fann_sys::fann_get_training_algorithm(self.raw) };
        TrainAlgorithm::from_fann_train_enum(ft_enum)
    }

    /// Set the algorithm to be used for training.
    pub fn set_train_algorithm(&mut self, ta: TrainAlgorithm) {
        let ft_enum = ta.to_fann_train_enum();
        unsafe { fann_sys::fann_set_training_algorithm(self.raw, ft_enum) }
    }

    /// Get the learning rate, which is used to determine how aggressive training should be (not
    /// used by the RPROP algorithm). The default is 0.7.
    pub fn get_learning_rate(&self) -> c_float {
        unsafe { fann_sys::fann_get_learning_rate(self.raw) }
    }

    /// Set the learning rate, which is used to determine how aggressive training should be (not
    /// used by the RPROP algorithm). The default is 0.7.
    pub fn set_learning_rate(&mut self, learning_rate: c_float) {
        unsafe { fann_sys::fann_set_learning_rate(self.raw, learning_rate) }
    }

    /// Get the learning momentum used in incremental training. It is recommended to use a value
    /// between 0.0 and 1.0. The default is 1.0.
    pub fn get_learning_momentum(&self) -> c_float {
        unsafe { fann_sys::fann_get_learning_momentum(self.raw) }
    }

    /// Set the learning momentum used in incremental training. It is recommended to use a value
    /// between 0.0 and 1.0. The default is 1.0.
    pub fn set_learning_momentum(&mut self, learning_momentum: c_float) {
        unsafe { fann_sys::fann_set_learning_momentum(self.raw, learning_momentum) }
    }
}

impl Drop for Fann {
    fn drop(&mut self) {
        unsafe { fann_sys::fann_destroy(self.raw); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.2;

    #[test]
    fn test_tutorial() {
        let max_epochs = 500000;
        let epochs_between_reports = 1000;
        let desired_error = 0.001;
        let mut fann = Fann::new(&[2, 3, 1]).unwrap();
        fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
        fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
        fann.train_on_file("test_files/xor.data",
                           max_epochs,
                           epochs_between_reports,
                           desired_error).unwrap();
        assert!(EPSILON > ( 1.0 - fann.run(&[-1.0,  1.0]).unwrap()[0]).abs());
        assert!(EPSILON > ( 1.0 - fann.run(&[ 1.0, -1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (-1.0 - fann.run(&[ 1.0,  1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (-1.0 - fann.run(&[-1.0, -1.0]).unwrap()[0]).abs());
    }
}
