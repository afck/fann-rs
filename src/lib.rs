//! A Rust wrapper for the Fast Artificial Neural Network library.
//!
//! A new neural network with random weights can be created with the `Fann::new` method, or, for
//! different network topologies, with its variants `Fann::new_sparse` and `Fann::new_shortcut`.
//! Existing neural networks can be saved to and loaded from files.
//!
//! Similarly, training data sets can be loaded from and saved to human-readable files, or training
//! data can be provided directly to the network as slices of floating point numbers.
//!
//! Example:
//!
//! ```
//! extern crate fann;
//! use fann::{ActivationFunc, Fann, TrainAlgorithm, QuickpropParams};
//!
//! fn main() {
//!    // Create a new network with two input neurons, a hidden layer with three neurons, and one
//!    // output neuron.
//!    let mut fann = Fann::new(&[2, 3, 1]).unwrap();
//!    // Configure the activation functions for the hidden and output neurons.
//!    fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
//!    fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
//!    // Use the Quickprop learning algorithm, with default parameters.
//!    // (Otherwise, Rprop would be used.)
//!    fann.set_train_algorithm(TrainAlgorithm::Quickprop(Default::default()));
//!    // Train for up to 500000 epochs, displaying progress information after intervals of 1000
//!    // epochs. Stop when the network's error on the training data drops to 0.001.
//!    let max_epochs = 500000;
//!    let epochs_between_reports = 1000;
//!    let desired_error = 0.001;
//!    // Train directly on data loaded from the file "xor.data".
//!    fann.on_file("test_files/xor.data")
//!        .with_reports(epochs_between_reports)
//!        .train(max_epochs, desired_error).unwrap();
//!    // The network now approximates the XOR problem:
//!    assert!(fann.run(&[-1.0,  1.0]).unwrap()[0] > 0.9);
//!    assert!(fann.run(&[ 1.0, -1.0]).unwrap()[0] > 0.9);
//!    assert!(fann.run(&[ 1.0,  1.0]).unwrap()[0] < 0.1);
//!    assert!(fann.run(&[-1.0, -1.0]).unwrap()[0] < 0.1);
//! }
//! ```
//!
//! FANN also supports cascade training, where the network's topology is changed during training by
//! adding additional neurons:
//!
//! ```
//! extern crate fann;
//! use fann::{ActivationFunc, CascadeParams, Fann};
//!
//! fn main() {
//!    // Create a new network with two input neurons and one output neuron.
//!    let mut fann = Fann::new_shortcut(&[2, 1]).unwrap();
//!    // Use the default cascade training parameters, but a higher weight multiplier:
//!    fann.set_cascade_params(&CascadeParams {
//!                                 weight_multiplier: 0.6,
//!                                 ..CascadeParams::default()
//!                             });
//!    // Add up to 50 neurons, displaying progress information after each.
//!    // Stop when the network's error on the training data drops to 0.001.
//!    let max_neurons = 50;
//!    let neurons_between_reports = 1;
//!    let desired_error = 0.001;
//!    // Train directly on data loaded from the file "xor.data".
//!    fann.on_file("test_files/xor.data")
//!        .with_reports(neurons_between_reports)
//!        .cascade()
//!        .train(max_neurons, desired_error).unwrap();
//!    // The network now approximates the XOR problem:
//!    assert!(fann.run(&[-1.0,  1.0]).unwrap()[0] > 0.9);
//!    assert!(fann.run(&[ 1.0, -1.0]).unwrap()[0] > 0.9);
//!    assert!(fann.run(&[ 1.0,  1.0]).unwrap()[0] < 0.1);
//!    assert!(fann.run(&[-1.0, -1.0]).unwrap()[0] < 0.1);
//! }
//! ```

extern crate fann_sys;
extern crate libc;

use fann_sys::*;
use libc::{c_float, c_int, c_uint};
use std::cell::RefCell;
use std::ffi::CString;
use std::mem::{forget, transmute};
use std::path::Path;
use std::ptr::{copy_nonoverlapping, null_mut};

pub use activation_func::ActivationFunc;
pub use cascade_params::CascadeParams;
pub use error::{FannError, FannErrorType, FannResult};
pub use error_func::ErrorFunc;
pub use net_type::NetType;
pub use stop_func::StopFunc;
pub use train_algorithm::TrainAlgorithm;
pub use train_algorithm::{BatchParams, IncrementalParams, QuickpropParams, RpropParams};
pub use train_data::TrainData;

mod activation_func;
mod cascade_params;
mod error;
mod error_func;
mod net_type;
mod stop_func;
mod train_algorithm;
mod train_data;

/// The type of weights, inputs and outputs in a neural network. It is defined as `c_float` by
/// default, and as `c_double` if the `double` feature is configured.
pub type FannType = fann_type;

pub type Connection = fann_connection;

/// Convert a path to a `CString`.
fn to_filename<P: AsRef<Path>>(path: P) -> Result<CString, FannError> {
    match path.as_ref().to_str().map(CString::new) {
        None => Err(FannError {
            error_type: FannErrorType::CantOpenTdR,
            error_str: "File name contains invalid unicode characters".to_owned(),
        }),
        Some(Err(e)) => Err(FannError {
            error_type: FannErrorType::CantOpenTdR,
            error_str: format!(
                "File name contains a nul byte at position {}",
                e.nul_position()
            ),
        }),
        Some(Ok(cs)) => Ok(cs),
    }
}

/// Either an owned or a borrowed `TrainData`.
enum CurrentTrainData<'a> {
    Own(FannResult<TrainData>),
    Ref(&'a TrainData),
}

// Thread-local container for a pointer to the current FannTrainer.
// This is necessary because the raw fann_train_on_data_with_callback C function takes a function
// pointer and not a closure. So instead of the user-supplied function we pass a function to it
// which will call the callback stored in the trainer.
// The 'static lifetime is a lie! But the trainer lives longer than the train method runs, and
// afterwards resets this pointer to null again.
thread_local!(static TRAINER: RefCell<*mut FannTrainer<'static>> = RefCell::new(null_mut()));

#[derive(Clone, Copy, Debug)]
pub enum CallbackResult {
    Stop,
    Continue,
}

impl CallbackResult {
    pub fn stop_if(condition: bool) -> CallbackResult {
        if condition {
            CallbackResult::Stop
        } else {
            CallbackResult::Continue
        }
    }
}

/// A training configuration. Create this with `Fann::on_data` or `Fann::on_file` and run the
/// training with `train`.
pub struct FannTrainer<'a> {
    fann: &'a mut Fann,
    cur_data: CurrentTrainData<'a>,
    callback: Option<&'a dyn Fn(&Fann, &TrainData, c_uint) -> CallbackResult>,
    interval: c_uint,
    cascade: bool,
}

impl<'a> FannTrainer<'a> {
    fn with_data<'b>(fann: &'b mut Fann, data: &'b TrainData) -> FannTrainer<'b> {
        FannTrainer {
            fann,
            cur_data: CurrentTrainData::Ref(data),
            callback: None,
            interval: 0,
            cascade: false,
        }
    }

    fn with_file<P: AsRef<Path>>(fann: &mut Fann, path: P) -> FannTrainer {
        FannTrainer {
            fann,
            cur_data: CurrentTrainData::Own(TrainData::from_file(path)),
            callback: None,
            interval: 0,
            cascade: false,
        }
    }

    /// Activates printing reports periodically. Between two reports, `interval` neurons are added
    /// (for cascade training) or training goes on for `interval` epochs (otherwise).
    pub fn with_reports(self, interval: c_uint) -> FannTrainer<'a> {
        FannTrainer { interval, ..self }
    }

    /// Configures a callback to be called periodically during training. Every `interval` epochs
    /// (for regular training) or every time `interval` new neurons have been added (for cascade
    /// training), the callback runs. It receives as arguments:
    ///
    /// * a reference to the current `Fann`,
    /// * a reference to the training data,
    /// * the number of steps (added neurons or epochs) taken so far.
    pub fn with_callback(
        self,
        interval: c_uint,
        callback: &'a dyn Fn(&Fann, &TrainData, c_uint) -> CallbackResult,
    ) -> FannTrainer<'a> {
        FannTrainer {
            callback: Some(callback),
            interval,
            ..self
        }
    }

    /// Use the Cascade2 algorithm: This adds neurons to the neural network while training, starting
    /// with an ANN without any hidden layers. The network should use shortcut connections, so it
    /// needs to be created like this:
    ///
    /// ```
    /// let td = fann::TrainData::from_file("test_files/xor.data").unwrap();
    /// let fann = fann::Fann::new_shortcut(&[td.num_input(), td.num_output()]).unwrap();
    /// ```
    pub fn cascade(self) -> FannTrainer<'a> {
        FannTrainer {
            cascade: true,
            ..self
        }
    }

    extern "C" fn raw_callback(
        ann: *mut fann,
        td: *mut fann_train_data,
        _: c_uint,
        _: c_uint,
        _: c_float,
        steps: c_uint,
    ) -> c_int {
        // TODO: This is an ugly hack - find better ways to solve the following issues:
        // * The C callback is not a closure, so it cannot access the user-supplied argument.
        //   https://aatch.github.io/blog/2015/01/17/unboxed-closures-and-ffi-callbacks doesn't
        //   work here because the C callback doesn't take a user-defined pointer as an argument.
        //   Instead, we store a pointer to the FannTrainer, which contains a fat pointer to the
        //   callback, in a thread-local variable that is accessed by the raw callback.
        // * The lifetime isn't known at the point where the thread-local variable is declared, so
        //   we just use 'static and transmute the pointer!
        // * The C callback is only given pointers to the raw structs, not to self and data. We
        //   read these from the tread-local variable, too, and assert that they correspond to the
        //   given raw structs.
        // * https://github.com/rust-lang/rust/issues/24010 seems to make it impossible to define a
        //   trait that would act as a shortcut for Fn(...) -> CallbackResult.
        match TRAINER.with(|cell| unsafe {
            let trainer = *cell.borrow();
            let data = (*trainer).get_data().unwrap();
            assert_eq!(ann, (*trainer).fann.raw);
            assert_eq!(td, data.get_raw());
            let callback = (*trainer).callback.unwrap();
            callback((*trainer).fann, data, steps)
        }) {
            CallbackResult::Stop => -1,
            CallbackResult::Continue => 0,
        }
    }

    fn get_data(&'a self) -> FannResult<&'a TrainData> {
        match self.cur_data {
            CurrentTrainData::Ref(data) => Ok(data),
            CurrentTrainData::Own(ref result) => result.as_ref().map_err(FannError::clone),
        }
    }

    /// Train the network until either the mean square error drops below the `desired_error`, or
    /// the maximum number of steps is reached. If cascade training is activated, `max_steps`
    /// refers to the number of neurons that are added, otherwise it is the maximum number of
    /// training epochs.
    // Clippy's suggestion fails: https://github.com/rust-lang-nursery/rust-clippy/issues/3340
    #[cfg_attr(feature = "cargo-clippy", allow(useless_transmute))]
    pub fn train(&mut self, max_steps: c_uint, desired_error: c_float) -> FannResult<()> {
        unsafe {
            let raw_data = self.get_data()?.get_raw();
            if self.callback.is_some() {
                TRAINER.with(|cell| *cell.borrow_mut() = transmute(&mut *self));
                fann_set_callback(self.fann.raw, Some(FannTrainer::raw_callback));
            }
            let raw_train_fn = if self.cascade {
                fann_cascadetrain_on_data
            } else {
                fann_train_on_data
            };
            raw_train_fn(
                self.fann.raw,
                raw_data,
                max_steps,
                self.interval,
                desired_error,
            );
            if self.callback.is_some() {
                fann_set_callback(self.fann.raw, None);
                TRAINER.with(|cell| *cell.borrow_mut() = null_mut());
            }
            FannError::check_no_error(self.fann.raw as *mut fann_error)
        }
    }
}

pub struct Fann {
    // We don't consider setting and clearing the error string and number a mutation, and every
    // method should leave these fields cleared, either because it succeeded or because it read the
    // fields and returned the corresponding error.
    // We also don't consider writing the output data a mutation, as we don't provide access to it
    // and copy it before returning it.
    raw: *mut fann,
}

impl Fann {
    unsafe fn from_raw(raw: *mut fann) -> FannResult<Fann> {
        FannError::check_no_error(raw as *mut fann_error)?;
        Ok(Fann { raw })
    }

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
    pub fn new(layers: &[c_uint]) -> FannResult<Fann> {
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
    pub fn new_sparse(connection_rate: c_float, layers: &[c_uint]) -> FannResult<Fann> {
        unsafe {
            Fann::from_raw(fann_create_sparse_array(
                connection_rate,
                layers.len() as c_uint,
                layers.as_ptr(),
            ))
        }
    }

    /// Create a neural network which has shortcut connections, i. e. it doesn't connect only each
    /// layer to its successor, but every layer with every later layer: Each neuron has connections
    /// to all neurons in all subsequent layers.
    pub fn new_shortcut(layers: &[c_uint]) -> FannResult<Fann> {
        unsafe {
            Fann::from_raw(fann_create_shortcut_array(
                layers.len() as c_uint,
                layers.as_ptr(),
            ))
        }
    }

    /// Read a neural network from a file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> FannResult<Fann> {
        let filename = to_filename(path)?;
        unsafe { Fann::from_raw(fann_create_from_file(filename.as_ptr())) }
    }

    /// Save the network to a configuration file.
    ///
    /// The file will contain all information about the neural network, except parameters generated
    /// during training, like mean square error and the bit fail limit.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> FannResult<()> {
        let filename = to_filename(path)?;
        unsafe {
            let result = fann_save(self.raw, filename.as_ptr());
            FannError::check_zero(result, self.raw as *mut fann_error, "Error saving network")
        }
    }

    /// Give each connection a random weight between `min_weight` and `max_weight`.
    ///
    /// By default, weights in a new network are random between -0.1 and 0.1.
    pub fn randomize_weights(&mut self, min_weight: FannType, max_weight: FannType) {
        unsafe { fann_randomize_weights(self.raw, min_weight, max_weight) }
    }

    /// Initialize the weights using Widrow & Nguyen's algorithm.
    ///
    /// The algorithm developed by Derrick Nguyen and Bernard Widrow sets the weight in a way that
    /// can speed up training with the given training data. This technique is not always successful
    /// and in some cases can even be less efficient that a purely random initialization.
    pub fn init_weights(&mut self, train_data: &TrainData) {
        unsafe { fann_init_weights(self.raw, train_data.get_raw()) }
    }

    /// Print the connections of the network in a compact matrix, for easy viewing of its
    /// internals.
    ///
    /// The output on a small (2 2 1) network trained on the xor problem:
    ///
    /// ```text
    /// Layer / Neuron 012345
    /// L   1 / N    3 BBa...
    /// L   1 / N    4 BBA...
    /// L   1 / N    5 ......
    /// L   2 / N    6 ...BBA
    /// L   2 / N    7 ......
    /// ```
    ///
    /// This network has five real neurons and two bias neurons. This gives a total of seven
    /// neurons named from 0 to 6. The connections between these neurons can be seen in the matrix.
    /// "." is a place where there is no connection, while a character tells how strong the
    /// connection is on a scale from a-z. The two real neurons in the hidden layer (neuron 3 and 4
    /// in layer 1) have connections from the three neurons in the previous layer as is visible in
    /// the first two lines. The output neuron 6 has connections from the three neurons in the
    /// hidden layer 3 - 5 as is visible in the fourth line.
    ///
    /// To simplify the matrix output neurons are not visible as neurons that connections can come
    /// from, and input and bias neurons are not visible as neurons that connections can go to.
    pub fn print_connections(&self) {
        unsafe { fann_print_connections(self.raw) }
    }

    /// Print all parameters and options of the network.
    pub fn print_parameters(&self) {
        unsafe { fann_print_parameters(self.raw) }
    }

    /// Return an `Err` if the size of the slice does not match the number of input neurons,
    /// otherwise `Ok(())`.
    fn check_input_size(&self, input: &[FannType]) -> FannResult<()> {
        let num_input = self.get_num_input() as usize;
        if input.len() == num_input {
            Ok(())
        } else {
            Err(FannError {
                error_type: FannErrorType::IndexOutOfBound,
                error_str: format!(
                    "Input has length {}, but there are {} input neurons",
                    input.len(),
                    num_input
                ),
            })
        }
    }

    /// Return an `Err` if the size of the slice does not match the number of output neurons,
    /// otherwise `Ok(())`.
    fn check_output_size(&self, output: &[FannType]) -> FannResult<()> {
        let num_output = self.get_num_output() as usize;
        if output.len() == num_output {
            Ok(())
        } else {
            Err(FannError {
                error_type: FannErrorType::IndexOutOfBound,
                error_str: format!(
                    "Output has length {}, but there are {} output neurons",
                    output.len(),
                    num_output
                ),
            })
        }
    }

    /// Train with a single pair of input and output. This is always incremental training (see
    /// `TrainAlg`), since only one pattern is presented.
    pub fn train(&mut self, input: &[FannType], desired_output: &[FannType]) -> FannResult<()> {
        unsafe {
            self.check_input_size(input)?;
            self.check_output_size(desired_output)?;
            fann_train(self.raw, input.as_ptr(), desired_output.as_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)?;
        }
        Ok(())
    }

    /// Create a training configuration for the given data set.
    pub fn on_data<'a>(&'a mut self, data: &'a TrainData) -> FannTrainer<'a> {
        FannTrainer::with_data(self, data)
    }

    /// Create a training configuration, reading the training data from the given file.
    pub fn on_file<P: AsRef<Path>>(&mut self, path: P) -> FannTrainer {
        FannTrainer::with_file(self, path)
    }

    /// Train one epoch with a set of training data, i. e. each sample from the training data is
    /// considered exactly once.
    ///
    /// Returns the mean square error as it is calculated either before or during the actual
    /// training. This is not the actual MSE after the training epoch, but since calculating this
    /// will require to go through the entire training set once more, it is more than adequate to
    /// use this value during training.
    pub fn train_epoch(&mut self, data: &TrainData) -> FannResult<c_float> {
        unsafe {
            let mse = fann_train_epoch(self.raw, data.get_raw());
            FannError::check_no_error(self.raw as *mut fann_error)?;
            Ok(mse)
        }
    }

    /// Test with a single pair of input and output. This operation updates the mean square error
    /// but does not change the network.
    ///
    /// Returns the actual output of the network.
    pub fn test(
        &mut self,
        input: &[FannType],
        desired_output: &[FannType],
    ) -> FannResult<Vec<FannType>> {
        self.check_input_size(input)?;
        self.check_output_size(desired_output)?;
        let num_output = self.get_num_output() as usize;
        let mut result = Vec::with_capacity(num_output);
        unsafe {
            let output = fann_test(self.raw, input.as_ptr(), desired_output.as_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)?;
            copy_nonoverlapping(output, result.as_mut_ptr(), num_output);
            result.set_len(num_output);
        }
        Ok(result)
    }

    /// Test with a training data set and calculate the mean square error.
    pub fn test_data(&mut self, data: &TrainData) -> FannResult<c_float> {
        unsafe {
            let mse = fann_test_data(self.raw, data.get_raw());
            FannError::check_no_error(self.raw as *mut fann_error)?;
            Ok(mse)
        }
    }

    /// Get the mean square error.
    pub fn get_mse(&self) -> c_float {
        unsafe { fann_get_MSE(self.raw) }
    }

    /// Get the number of fail bits, i. e. the number of neurons which differed from the desired
    /// output by more than the bit fail limit since the previous reset.
    pub fn get_bit_fail(&self) -> c_uint {
        unsafe { fann_get_bit_fail(self.raw) }
    }

    /// Reset the mean square error and bit fail count.
    pub fn reset_mse_and_bit_fail(&mut self) {
        unsafe {
            fann_reset_MSE(self.raw);
        }
    }

    /// Run the input through the neural network and returns the output. The length of the input
    /// must equal the number of input neurons and the length of the output will equal the number
    /// of output neurons.
    pub fn run(&self, input: &[FannType]) -> FannResult<Vec<FannType>> {
        self.check_input_size(input)?;
        let num_output = self.get_num_output() as usize;
        let mut result = Vec::with_capacity(num_output);
        unsafe {
            let output = fann_run(self.raw, input.as_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)?;
            copy_nonoverlapping(output, result.as_mut_ptr(), num_output);
            result.set_len(num_output);
        }
        Ok(result)
    }

    /// Get the number of input neurons.
    pub fn get_num_input(&self) -> c_uint {
        unsafe { fann_get_num_input(self.raw) }
    }

    /// Get the number of output neurons.
    pub fn get_num_output(&self) -> c_uint {
        unsafe { fann_get_num_output(self.raw) }
    }

    /// Get the total number of neurons, including the bias neurons.
    ///
    /// E. g. a 2-4-2 network has 3 + 5 + 2 = 10 neurons (because two layers have bias neurons).
    pub fn get_total_neurons(&self) -> c_uint {
        unsafe { fann_get_total_neurons(self.raw) }
    }

    /// Get the total number of connections.
    pub fn get_total_connections(&self) -> c_uint {
        unsafe { fann_get_total_connections(self.raw) }
    }

    /// Get the type of the neural network.
    pub fn get_network_type(&self) -> NetType {
        let nt_enum = unsafe { fann_get_network_type(self.raw) };
        NetType::from_nettype_enum(nt_enum)
    }

    /// Get the connection rate used when the network was created.
    pub fn get_connection_rate(&self) -> c_float {
        unsafe { fann_get_connection_rate(self.raw) }
    }

    /// Get the number of layers in the network.
    pub fn get_num_layers(&self) -> c_uint {
        unsafe { fann_get_num_layers(self.raw) }
    }

    /// Get the number of neurons in each layer of the network.
    pub fn get_layer_sizes(&self) -> Vec<c_uint> {
        let num_layers = self.get_num_layers() as usize;
        let mut result = Vec::with_capacity(num_layers);
        unsafe {
            fann_get_layer_array(self.raw, result.as_mut_ptr());
            result.set_len(num_layers);
        }
        result
    }

    /// Get the number of bias neurons in each layer of the network.
    pub fn get_bias_counts(&self) -> Vec<c_uint> {
        let num_layers = self.get_num_layers() as usize;
        let mut result = Vec::with_capacity(num_layers);
        unsafe {
            fann_get_bias_array(self.raw, result.as_mut_ptr());
            result.set_len(num_layers);
        }
        result
    }

    /// Get a list of all connections in the network.
    pub fn get_connections(&self) -> Vec<Connection> {
        let total = self.get_total_connections() as usize;
        let mut result = Vec::with_capacity(total);
        unsafe {
            fann_get_connection_array(self.raw, result.as_mut_ptr());
            result.set_len(total);
        }
        result
    }

    /// Set the weights of all given connections.
    ///
    /// Connections that don't already exist are ignored.
    pub fn set_connections<'a, I: IntoIterator<Item = &'a Connection>>(&mut self, connections: I) {
        for c in connections {
            self.set_weight(c.from_neuron, c.to_neuron, c.weight);
        }
    }

    /// Set the weight of the given connection.
    pub fn set_weight(&mut self, from_neuron: c_uint, to_neuron: c_uint, weight: FannType) {
        unsafe { fann_set_weight(self.raw, from_neuron, to_neuron, weight) }
    }

    /// Get the activation function for neuron number `neuron` in layer number `layer`, counting
    /// the input layer as number 0. Input layer neurons do not have an activation function, so
    /// `layer` must be at least 1.
    pub fn get_activation_func(&self, layer: c_int, neuron: c_int) -> FannResult<ActivationFunc> {
        let af_enum = unsafe { fann_get_activation_function(self.raw, layer, neuron) };
        unsafe { FannError::check_no_error(self.raw as *mut fann_error)? };
        ActivationFunc::from_fann_activationfunc_enum(af_enum)
    }

    /// Set the activation function for neuron number `neuron` in layer number `layer`, counting
    /// the input layer as number 0. Input layer neurons do not have an activation function, so
    /// `layer` must be at least 1.
    pub fn set_activation_func(&mut self, af: ActivationFunc, layer: c_int, neuron: c_int) {
        let af_enum = af.to_fann_activationfunc_enum();
        unsafe { fann_set_activation_function(self.raw, af_enum, layer, neuron) }
    }

    /// Set the activation function for all hidden layers.
    pub fn set_activation_func_hidden(&mut self, activation_func: ActivationFunc) {
        unsafe {
            let af_enum = activation_func.to_fann_activationfunc_enum();
            fann_set_activation_function_hidden(self.raw, af_enum);
        }
    }

    /// Set the activation function for the output layer.
    pub fn set_activation_func_output(&mut self, activation_func: ActivationFunc) {
        unsafe {
            let af_enum = activation_func.to_fann_activationfunc_enum();
            fann_set_activation_function_output(self.raw, af_enum)
        }
    }

    /// Get the activation steepness for neuron number `neuron` in layer number `layer`.
    #[cfg_attr(feature = "cargo-clippy", allow(float_cmp))]
    pub fn get_activation_steepness(&self, layer: c_int, neuron: c_int) -> Option<FannType> {
        let steepness = unsafe { fann_get_activation_steepness(self.raw, layer, neuron) };
        // This returns exactly -1 if the neuron is not defined.
        if steepness == -1.0 {
            return None;
        }
        Some(steepness)
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
    pub fn set_activation_steepness(&self, steepness: FannType, layer: c_int, neuron: c_int) {
        unsafe { fann_set_activation_steepness(self.raw, steepness, layer, neuron) }
    }

    /// Set the activation steepness for layer number `layer`.
    pub fn set_activation_steepness_layer(&self, steepness: FannType, layer: c_int) {
        unsafe { fann_set_activation_steepness_layer(self.raw, steepness, layer) }
    }

    /// Set the activation steepness for all hidden layers.
    pub fn set_activation_steepness_hidden(&self, steepness: FannType) {
        unsafe { fann_set_activation_steepness_hidden(self.raw, steepness) }
    }

    /// Set the activation steepness for the output layer.
    pub fn set_activation_steepness_output(&self, steepness: FannType) {
        unsafe { fann_set_activation_steepness_output(self.raw, steepness) }
    }

    /// Get the error function used during training.
    pub fn get_error_func(&self) -> ErrorFunc {
        let ef_enum = unsafe { fann_get_train_error_function(self.raw) };
        ErrorFunc::from_errorfunc_enum(ef_enum)
    }

    /// Set the error function used during training.
    ///
    /// The default is `Tanh`.
    pub fn set_error_func(&mut self, ef: ErrorFunc) {
        let ef_enum = ef.to_errorfunc_enum();
        unsafe { fann_set_train_error_function(self.raw, ef_enum) }
    }

    /// Get the stop criterion for training.
    pub fn get_stop_func(&self) -> StopFunc {
        let sf_enum = unsafe { fann_get_train_stop_function(self.raw) };
        StopFunc::from_stopfunc_enum(sf_enum)
    }

    /// Set the stop criterion for training.
    ///
    /// The default is `Mse`.
    pub fn set_stop_func(&mut self, sf: StopFunc) {
        let sf_enum = sf.to_stopfunc_enum();
        unsafe { fann_set_train_stop_function(self.raw, sf_enum) }
    }

    /// Get the bit fail limit.
    pub fn get_bit_fail_limit(&self) -> FannType {
        unsafe { fann_get_bit_fail_limit(self.raw) }
    }

    /// Set the bit fail limit.
    ///
    /// Each output neuron value that differs from the desired output by more than the bit fail
    /// limit is counted as a failed bit.
    pub fn set_bit_fail_limit(&mut self, bit_fail_limit: FannType) {
        unsafe { fann_set_bit_fail_limit(self.raw, bit_fail_limit) }
    }

    /// Get cascade training parameters.
    pub fn get_cascade_params(&self) -> CascadeParams {
        unsafe {
            let num_af = fann_get_cascade_activation_functions_count(self.raw) as usize;
            let af_enum_ptr = fann_get_cascade_activation_functions(self.raw);
            let af_enums = Vec::from_raw_parts(af_enum_ptr, num_af, num_af);
            let activation_functions = af_enums
                .iter()
                .map(|&af_enum| ActivationFunc::from_fann_activationfunc_enum(af_enum).unwrap())
                .collect();
            forget(af_enums);
            let num_st = fann_get_cascade_activation_steepnesses_count(self.raw) as usize;
            let mut activation_steepnesses = Vec::with_capacity(num_st);
            let st_ptr = fann_get_cascade_activation_steepnesses(self.raw);
            copy_nonoverlapping(st_ptr, activation_steepnesses.as_mut_ptr(), num_st);
            activation_steepnesses.set_len(num_st);
            CascadeParams {
                output_change_fraction: fann_get_cascade_output_change_fraction(self.raw),
                output_stagnation_epochs: fann_get_cascade_output_stagnation_epochs(self.raw),
                candidate_change_fraction: fann_get_cascade_candidate_change_fraction(self.raw),
                candidate_stagnation_epochs: fann_get_cascade_candidate_stagnation_epochs(self.raw),
                candidate_limit: fann_get_cascade_candidate_limit(self.raw),
                weight_multiplier: fann_get_cascade_weight_multiplier(self.raw),
                max_out_epochs: fann_get_cascade_max_out_epochs(self.raw),
                max_cand_epochs: fann_get_cascade_max_cand_epochs(self.raw),
                activation_functions,
                activation_steepnesses,
                num_candidate_groups: fann_get_cascade_num_candidate_groups(self.raw),
            }
        }
    }

    /// Set cascade training parameters.
    pub fn set_cascade_params(&mut self, params: &CascadeParams) {
        let af_enums: Vec<_> = params
            .activation_functions
            .iter()
            .map(|af| af.to_fann_activationfunc_enum())
            .collect();
        unsafe {
            fann_set_cascade_output_change_fraction(self.raw, params.output_change_fraction);
            fann_set_cascade_output_stagnation_epochs(self.raw, params.output_stagnation_epochs);
            fann_set_cascade_candidate_change_fraction(self.raw, params.candidate_change_fraction);
            fann_set_cascade_candidate_stagnation_epochs(
                self.raw,
                params.candidate_stagnation_epochs,
            );
            fann_set_cascade_candidate_limit(self.raw, params.candidate_limit);
            fann_set_cascade_weight_multiplier(self.raw, params.weight_multiplier);
            fann_set_cascade_max_out_epochs(self.raw, params.max_out_epochs);
            fann_set_cascade_max_cand_epochs(self.raw, params.max_cand_epochs);
            fann_set_cascade_activation_functions(
                self.raw,
                af_enums.as_ptr(),
                af_enums.len() as c_uint,
            );
            fann_set_cascade_activation_steepnesses(
                self.raw,
                params.activation_steepnesses.as_ptr(),
                params.activation_steepnesses.len() as c_uint,
            );
            fann_set_cascade_num_candidate_groups(self.raw, params.num_candidate_groups);
        }
    }

    /// Get the currently configured training algorithm.
    pub fn get_train_algorithm(&self) -> TrainAlgorithm {
        let ft_enum = unsafe { fann_get_training_algorithm(self.raw) };
        match ft_enum {
            FANN_TRAIN_INCREMENTAL => unsafe {
                TrainAlgorithm::Incremental(IncrementalParams {
                    learning_momentum: fann_get_learning_momentum(self.raw),
                    learning_rate: fann_get_learning_rate(self.raw),
                })
            },
            FANN_TRAIN_BATCH => unsafe {
                TrainAlgorithm::Batch(BatchParams {
                    learning_rate: fann_get_learning_rate(self.raw),
                })
            },
            FANN_TRAIN_RPROP => unsafe {
                TrainAlgorithm::Rprop(RpropParams {
                    decrease_factor: fann_get_rprop_decrease_factor(self.raw),
                    increase_factor: fann_get_rprop_increase_factor(self.raw),
                    delta_min: fann_get_rprop_delta_min(self.raw),
                    delta_max: fann_get_rprop_delta_max(self.raw),
                    delta_zero: fann_get_rprop_delta_zero(self.raw),
                })
            },
            FANN_TRAIN_QUICKPROP => unsafe {
                TrainAlgorithm::Quickprop(QuickpropParams {
                    decay: fann_get_quickprop_decay(self.raw),
                    mu: fann_get_quickprop_mu(self.raw),
                    learning_rate: fann_get_learning_rate(self.raw),
                })
            },
        }
    }

    /// Set the algorithm to be used for training.
    pub fn set_train_algorithm(&mut self, ta: TrainAlgorithm) {
        match ta {
            TrainAlgorithm::Incremental(params) => unsafe {
                fann_set_training_algorithm(self.raw, FANN_TRAIN_INCREMENTAL);
                fann_set_learning_momentum(self.raw, params.learning_momentum);
                fann_set_learning_rate(self.raw, params.learning_rate);
            },
            TrainAlgorithm::Batch(params) => unsafe {
                fann_set_training_algorithm(self.raw, FANN_TRAIN_BATCH);
                fann_set_learning_rate(self.raw, params.learning_rate);
            },
            TrainAlgorithm::Rprop(params) => unsafe {
                fann_set_training_algorithm(self.raw, FANN_TRAIN_RPROP);
                fann_set_rprop_decrease_factor(self.raw, params.decrease_factor);
                fann_set_rprop_increase_factor(self.raw, params.increase_factor);
                fann_set_rprop_delta_min(self.raw, params.delta_min);
                fann_set_rprop_delta_max(self.raw, params.delta_max);
                fann_set_rprop_delta_zero(self.raw, params.delta_zero);
            },
            TrainAlgorithm::Quickprop(params) => unsafe {
                fann_set_training_algorithm(self.raw, FANN_TRAIN_QUICKPROP);
                fann_set_quickprop_decay(self.raw, params.decay);
                fann_set_quickprop_mu(self.raw, params.mu);
                fann_set_learning_rate(self.raw, params.learning_rate);
            },
        }
    }

    /// Calculate input scaling parameters for future use based on the given training data.
    pub fn set_input_scaling_params(
        &mut self,
        data: &TrainData,
        new_input_min: c_float,
        new_input_max: c_float,
    ) -> FannResult<()> {
        unsafe {
            let result = fann_set_input_scaling_params(
                self.raw,
                data.get_raw(),
                new_input_min,
                new_input_max,
            );
            FannError::check_zero(
                result,
                self.raw as *mut fann_error,
                "Error calculating scaling parameters",
            )
        }
    }

    /// Calculate output scaling parameters for future use based on the given training data.
    pub fn set_output_scaling_params(
        &mut self,
        data: &TrainData,
        new_output_min: c_float,
        new_output_max: c_float,
    ) -> FannResult<()> {
        unsafe {
            let result = fann_set_output_scaling_params(
                self.raw,
                data.get_raw(),
                new_output_min,
                new_output_max,
            );
            FannError::check_zero(
                result,
                self.raw as *mut fann_error,
                "Error calculating scaling parameters",
            )
        }
    }

    /// Calculate scaling parameters for future use based on the given training data.
    pub fn set_scaling_params(
        &mut self,
        data: &TrainData,
        new_input_min: c_float,
        new_input_max: c_float,
        new_output_min: c_float,
        new_output_max: c_float,
    ) -> FannResult<()> {
        unsafe {
            let result = fann_set_scaling_params(
                self.raw,
                data.get_raw(),
                new_input_min,
                new_input_max,
                new_output_min,
                new_output_max,
            );
            FannError::check_zero(
                result,
                self.raw as *mut fann_error,
                "Error calculating scaling parameters",
            )
        }
    }

    /// Clear scaling parameters.
    pub fn clear_scaling_params(&mut self) -> FannResult<()> {
        unsafe {
            FannError::check_zero(
                fann_clear_scaling_params(self.raw),
                self.raw as *mut fann_error,
                "Error clearing scaling parameters",
            )
        }
    }

    /// Scale data in input vector before feeding it to the network, based on previously calculated
    /// parameters.
    pub fn scale_input(&self, input: &mut [FannType]) -> FannResult<()> {
        unsafe {
            fann_scale_input(self.raw, input.as_mut_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Scale data in output vector before feeding it to the network, based on previously calculated
    /// parameters.
    pub fn scale_output(&self, output: &mut [FannType]) -> FannResult<()> {
        unsafe {
            fann_scale_output(self.raw, output.as_mut_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Descale data in input vector after feeding it to the network, based on previously calculated
    /// parameters.
    pub fn descale_input(&self, input: &mut [FannType]) -> FannResult<()> {
        unsafe {
            fann_descale_input(self.raw, input.as_mut_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Descale data in output vector after getting it from the network, based on previously
    /// calculated parameters.
    pub fn descale_output(&self, output: &mut [FannType]) -> FannResult<()> {
        unsafe {
            fann_descale_output(self.raw, output.as_mut_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    // TODO: set_error_log: Always disable, due to different error handling?
    // TODO: save_to_fixed?
    // TODO: user_data methods?
}

impl Drop for Fann {
    fn drop(&mut self) {
        unsafe {
            fann_destroy(self.raw);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fann_sys;
    use libc::c_uint;
    use std::cell::RefCell;
    use std::ptr::null_mut;

    const EPSILON: FannType = 0.2;

    #[test]
    fn test_tutorial() {
        let max_epochs = 500_000;
        let desired_error = 0.0001;
        let mut fann = Fann::new(&[2, 3, 1]).unwrap();
        fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
        fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
        fann.on_file("test_files/xor.data")
            .train(max_epochs, desired_error)
            .unwrap();
        assert!(EPSILON > (1.0 - fann.run(&[-1.0, 1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (1.0 - fann.run(&[1.0, -1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (-1.0 - fann.run(&[1.0, 1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (-1.0 - fann.run(&[-1.0, -1.0]).unwrap()[0]).abs());
    }

    #[test]
    fn test_activation_func() {
        let mut fann = Fann::new(&[4, 3, 3, 1]).unwrap();
        // Don't print the expected errors:
        unsafe {
            fann_sys::fann_set_error_log(fann.raw as *mut fann_sys::fann_error, null_mut());
        }
        assert!(fann.get_activation_func(0, 1).is_err());
        assert!(fann.get_activation_func(4, 1).is_err());
        assert_eq!(
            Ok(ActivationFunc::SigmoidStepwise),
            fann.get_activation_func(2, 2)
        );
        fann.set_activation_func(ActivationFunc::Sin, 2, 2);
        assert_eq!(Ok(ActivationFunc::Sin), fann.get_activation_func(2, 2));
    }

    #[test]
    fn test_train_algorithm() {
        let mut fann = Fann::new(&[4, 3, 3, 1]).unwrap();
        assert_eq!(TrainAlgorithm::default(), fann.get_train_algorithm());
        let quickprop = TrainAlgorithm::Quickprop(QuickpropParams {
            decay: -0.0002,
            ..Default::default()
        });
        fann.set_train_algorithm(quickprop);
        assert_eq!(quickprop, fann.get_train_algorithm());
    }

    #[test]
    fn test_layer_sizes() {
        let fann = Fann::new(&[4, 3, 3, 1]).unwrap();
        assert_eq!(vec![4, 3, 3, 1], fann.get_layer_sizes());
        assert_eq!(vec![1, 1, 1, 0], fann.get_bias_counts());
    }

    #[test]
    fn test_get_set_connections() {
        let mut fann = Fann::new(&[1, 1]).unwrap();
        let connection = Connection {
            from_neuron: 1,
            to_neuron: 2,
            weight: 0.123,
        };
        fann.set_connections(&[connection]);
        assert_eq!(2, fann.get_total_connections()); // 2 because of the bias neuron in layer 0.
        assert_eq!(connection, fann.get_connections()[1]);
    }

    #[test]
    fn test_cascade_params() {
        let fann = Fann::new(&[1, 1]).unwrap();
        assert_eq!(CascadeParams::default(), fann.get_cascade_params());
    }

    #[test]
    fn test_train_data_from_callback() {
        let mut fann = Fann::new(&[2, 3, 1]).unwrap();
        fann.set_activation_func_hidden(ActivationFunc::SigmoidSymmetric);
        fann.set_activation_func_output(ActivationFunc::SigmoidSymmetric);
        let td = TrainData::from_callback(
            4,
            2,
            1,
            Box::new(|num| match num {
                0 => (vec![-1.0, 1.0], vec![1.0]),
                1 => (vec![1.0, -1.0], vec![1.0]),
                2 => (vec![-1.0, -1.0], vec![-1.0]),
                3 => (vec![1.0, 1.0], vec![-1.0]),
                _ => unreachable!(),
            }),
        ).unwrap();
        fann.on_data(&td).train(500_000, 0.0001).unwrap();
        assert!(EPSILON > (1.0 - fann.run(&[-1.0, 1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (1.0 - fann.run(&[1.0, -1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (-1.0 - fann.run(&[1.0, 1.0]).unwrap()[0]).abs());
        assert!(EPSILON > (-1.0 - fann.run(&[-1.0, -1.0]).unwrap()[0]).abs());
    }

    #[test]
    fn test_train_callback() {
        // Without a hidden layer, the XOR problem cannot be solved, so the training will only stop
        // when the callback says so.
        let mut fann = Fann::new(&[2, 1]).unwrap();
        fann.set_activation_func_output(ActivationFunc::LinearPiece);
        let xor_data = TrainData::from_file("test_files/xor.data").unwrap();
        let raw = fann.raw;
        let callback_epochs = RefCell::new(Vec::new());
        let cb = |fann: &Fann, train_data: &TrainData, epochs: c_uint| {
            assert_eq!(raw, fann.raw);
            unsafe {
                assert_eq!(xor_data.get_raw(), train_data.get_raw());
            }
            callback_epochs.borrow_mut().push(epochs);
            CallbackResult::stop_if(epochs == 40) // Stop after 40 epochs.
        };
        fann.on_data(&xor_data)
            .with_callback(10, &cb)
            .train(100, 0.1)
            .unwrap();
        // The interval was 10 epochs. Also, FANN always runs the callback after the first epoch.
        assert_eq!(vec![1, 10, 20, 30, 40], *callback_epochs.borrow());
    }
}
