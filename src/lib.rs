extern crate libc;
extern crate fann_sys;

use error::{FannError, FannErrorType};
use fann_sys::fann_activationfunc_enum::*;
use fann_sys::fann_type;
use libc::{c_float, c_uint};
use std::ffi::CString;
use std::path::Path;
use std::ptr::copy_nonoverlapping;

pub mod error;

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

/// Convert the path to a `CString`.
fn to_filename<P: AsRef<Path>>(path: P) -> Result<CString, FannError> {
    match path.as_ref().to_str().map(|s| CString::new(s)) {
        None => Err(FannError {
                    error_type: FannErrorType::CantOpenTdR,
                    error_str: "File name contains invalid unicode characters".to_string(),
                }),
        Some(Err(e)) => Err(FannError {
                            error_type: FannErrorType::CantOpenTdR,
                            error_str: format!("File name contains a nul byte at position {}",
                                               e.nul_position()),
                        }),
        Some(Ok(cs)) => Ok(cs),
    }
}

pub struct TrainData {
    raw: *mut fann_sys::fann_train_data,
}

impl TrainData {
    /// Read a file that stores training data.
    ///
    /// The file must be formatted like:
    ///
    /// ```text
    /// num_train_data num_input num_output
    /// inputdata separated by space
    /// outputdata separated by space
    /// .
    /// .
    /// .
    /// inputdata separated by space
    /// outputdata separated by space
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<TrainData, FannError> {
        let filename = try!(to_filename(path));
        unsafe {
            let raw = fann_sys::fann_read_train_from_file(filename.as_ptr());
            try!(FannError::check_no_error(raw as *mut fann_sys::fann_error));
            Ok(TrainData { raw: raw })
        }
    }

    /// Save the training data to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), FannError> {
        let filename = try!(to_filename(path));
        unsafe {
            let result = fann_sys::fann_save_train(self.raw, filename.as_ptr());
            try!(FannError::check_no_error(self.raw as *mut fann_sys::fann_error));
            if result == -1 {
                Err(FannError {
                    error_type: FannErrorType::CantSaveFile,
                    error_str: "Error saving training data".to_string(),
                })
            } else {
                Ok(())
            }
        }
    }

    /// Merge the given data sets into a new one.
    pub fn merge(data1: &TrainData, data2: &TrainData) -> Result<TrainData, FannError> {
        unsafe {
            let raw = fann_sys::fann_merge_train_data(data1.raw, data2.raw);
            try!(FannError::check_no_error(raw as *mut fann_sys::fann_error));
            Ok(TrainData { raw: raw })
        }
    }

    /// Create a subset of the training data, starting at the given positon and consisting of
    /// `length` samples.
    pub fn subset(&self, pos: c_uint, length: c_uint) -> Result<TrainData, FannError> {
        unsafe {
            let raw = fann_sys::fann_subset_train_data(self.raw, pos, length);
            try!(FannError::check_no_error(raw as *mut fann_sys::fann_error));
            Ok(TrainData { raw: raw })
        }
    }

    /// Return the number of training patterns in the data.
    pub fn length(&self) -> c_uint {
        unsafe { fann_sys::fann_length_train_data(self.raw) }
    }

    /// Return the number of input values in each training pattern.
    pub fn num_input(&self) -> c_uint {
        unsafe { fann_sys::fann_num_input_train_data(self.raw) }
    }

    /// Return the number of output values in each training pattern.
    pub fn num_output(&self) -> c_uint {
        unsafe { fann_sys::fann_num_output_train_data(self.raw) }
    }

    // TODO: from_callback
    // TODO: `scale` methods
    // TODO: save_to_fixed?

    /// Shuffle training data, randomizing the order. This is recommended for incremental training
    /// while it does not affect batch training.
    pub fn shuffle(&mut self) {
        unsafe { fann_sys::fann_shuffle_train_data(self.raw); }
    }
}

impl Clone for TrainData {
    fn clone(&self) -> TrainData {
        unsafe {
            let raw = fann_sys::fann_duplicate_train_data(self.raw);
            // TODO: Incorporate null check into check_no_error?
            if FannError::check_no_error(raw as *mut fann_sys::fann_error).is_err() {
                panic!("Unable to clone TrainData.");
            }
            TrainData { raw: raw }
        }
    }
}

impl Drop for TrainData {
    fn drop(&mut self) {
        unsafe { fann_sys::fann_destroy_train(self.raw); }
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
                                         data.raw,
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
            let mse = fann_sys::fann_train_epoch(self.raw, data.raw);
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
            let mse = fann_sys::fann_test_data(self.raw, data.raw);
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
    pub fn reset_mse(&mut self) {
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
            fann_sys::fann_set_activation_function_output(self.raw, af_enum);
        }
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
