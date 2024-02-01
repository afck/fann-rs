extern crate fann_sys;

use super::{to_filename, Fann};
use error::{FannError, FannErrorType, FannResult};
use fann_sys::*;
use libc::c_uint;
use std::cell::RefCell;
use std::path::Path;
use std::ptr::copy_nonoverlapping;

pub type TrainCallback = Fn(c_uint) -> (Vec<fann_type>, Vec<fann_type>);

// Thread-local container for user-supplied callback functions.
// This is necessary because the raw fann_create_train_from_callback C function takes a function
// pointer and not a closure. So instead of the user-supplied function we pass a function to it
// which will call the content of CALLBACK.
thread_local!(static CALLBACK: RefCell<Option<Box<TrainCallback>>> = RefCell::new(None));

pub struct TrainData {
    raw: *mut fann_train_data,
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
    pub fn from_file<P: AsRef<Path>>(path: P) -> FannResult<TrainData> {
        let filename = to_filename(path)?;
        unsafe {
            let raw = fann_read_train_from_file(filename.as_ptr());
            FannError::check_no_error(raw as *mut fann_error)?;
            Ok(TrainData { raw })
        }
    }

    /// Create training data using the given callback which for each number between `0` (included)
    /// and `num_data` (excluded) returns a pair of input and output vectors with `num_input` and
    /// `num_output` entries respectively.
    pub fn from_callback(
        num_data: c_uint,
        num_input: c_uint,
        num_output: c_uint,
        cb: Box<TrainCallback>,
    ) -> FannResult<TrainData> {
        extern "C" fn raw_callback(
            num: c_uint,
            num_input: c_uint,
            num_output: c_uint,
            input: *mut fann_type,
            output: *mut fann_type,
        ) {
            // Call the callback we stored in the thread-local container.
            let (in_vec, out_vec) = CALLBACK.with(|cell| cell.borrow().as_ref().unwrap()(num));
            // Make sure it returned data of the correct size, then copy the data.
            assert_eq!(in_vec.len(), num_input as usize);
            assert_eq!(out_vec.len(), num_output as usize);
            unsafe {
                copy_nonoverlapping(in_vec.as_ptr(), input, in_vec.len());
                copy_nonoverlapping(out_vec.as_ptr(), output, out_vec.len());
            }
        }
        unsafe {
            // Put the callback into the thread-local container.
            CALLBACK.with(|cell| *cell.borrow_mut() = Some(cb));
            let raw = fann_create_train_from_callback(
                num_data,
                num_input,
                num_output,
                Some(raw_callback),
            );
            // Remove it from the thread-local container to free the memory.
            CALLBACK.with(|cell| *cell.borrow_mut() = None);
            FannError::check_no_error(raw as *mut fann_error)?;
            Ok(TrainData { raw })
        }
    }

    /// Save the training data to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> FannResult<()> {
        let filename = to_filename(path)?;
        unsafe {
            let result = fann_save_train(self.raw, filename.as_ptr());
            FannError::check_no_error(self.raw as *mut fann_error)?;
            if result == -1 {
                Err(FannError {
                    error_type: FannErrorType::CantSaveFile,
                    error_str: "Error saving training data".to_owned(),
                })
            } else {
                Ok(())
            }
        }
    }

    /// Merge the given data sets into a new one.
    pub fn merge(data1: &TrainData, data2: &TrainData) -> FannResult<TrainData> {
        unsafe {
            let raw = fann_merge_train_data(data1.raw, data2.raw);
            FannError::check_no_error(raw as *mut fann_error)?;
            Ok(TrainData { raw })
        }
    }

    /// Create a subset of the training data, starting at the given positon and consisting of
    /// `length` samples.
    pub fn subset(&self, pos: c_uint, length: c_uint) -> FannResult<TrainData> {
        unsafe {
            let raw = fann_subset_train_data(self.raw, pos, length);
            FannError::check_no_error(raw as *mut fann_error)?;
            Ok(TrainData { raw })
        }
    }

    /// Return the number of training patterns in the data.
    pub fn length(&self) -> c_uint {
        unsafe { fann_length_train_data(self.raw) }
    }

    /// Return the number of input values in each training pattern.
    pub fn num_input(&self) -> c_uint {
        unsafe { fann_num_input_train_data(self.raw) }
    }

    /// Return the number of output values in each training pattern.
    pub fn num_output(&self) -> c_uint {
        unsafe { fann_num_output_train_data(self.raw) }
    }

    /// Scale input and output in the training data using the parameters previously calculated for
    /// the given network.
    pub fn scale_for(&mut self, fann: &Fann) -> FannResult<()> {
        unsafe {
            fann_scale_train(fann.raw, self.raw);
            FannError::check_no_error(fann.raw as *mut fann_error)?;
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Descale input and output in the training data using the parameters previously calculated for
    /// the given network.
    pub fn descale_for(&mut self, fann: &Fann) -> FannResult<()> {
        unsafe {
            fann_descale_train(fann.raw, self.raw);
            FannError::check_no_error(fann.raw as *mut fann_error)?;
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Scales the inputs in the training data to the specified range.
    pub fn scale_input(&mut self, new_min: fann_type, new_max: fann_type) -> FannResult<()> {
        unsafe {
            fann_scale_input_train_data(self.raw, new_min, new_max);
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Scales the outputs in the training data to the specified range.
    pub fn scale_output(&mut self, new_min: fann_type, new_max: fann_type) -> FannResult<()> {
        unsafe {
            fann_scale_output_train_data(self.raw, new_min, new_max);
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Scales the inputs and outputs in the training data to the specified range.
    pub fn scale(&mut self, new_min: fann_type, new_max: fann_type) -> FannResult<()> {
        unsafe {
            fann_scale_train_data(self.raw, new_min, new_max);
            FannError::check_no_error(self.raw as *mut fann_error)
        }
    }

    /// Shuffle training data, randomizing the order. This is recommended for incremental training
    /// while it does not affect batch training.
    pub fn shuffle(&mut self) {
        unsafe {
            fann_shuffle_train_data(self.raw);
        }
    }

    /// Get a pointer to the underlying raw `fann_train_data` structure.
    pub unsafe fn get_raw(&self) -> *mut fann_train_data {
        self.raw
    }

    // TODO: save_to_fixed?
}

impl Clone for TrainData {
    fn clone(&self) -> TrainData {
        unsafe {
            let raw = fann_duplicate_train_data(self.raw);
            if FannError::check_no_error(raw as *mut fann_error).is_err() {
                panic!("Unable to clone TrainData.");
            }
            TrainData { raw }
        }
    }
}

impl Drop for TrainData {
    fn drop(&mut self) {
        unsafe {
            fann_destroy_train(self.raw);
        }
    }
}
