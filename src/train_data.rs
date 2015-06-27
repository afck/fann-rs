extern crate fann_sys;

use error::{FannError, FannErrorType};
use libc::c_uint;
use std::ffi::CString;
use std::path::Path;

/// Convert a path to a `CString`.
pub fn to_filename<P: AsRef<Path>>(path: P) -> Result<CString, FannError> {
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
    // TODO: scale methods
    // TODO: save_to_fixed?

    /// Shuffle training data, randomizing the order. This is recommended for incremental training
    /// while it does not affect batch training.
    pub fn shuffle(&mut self) {
        unsafe { fann_sys::fann_shuffle_train_data(self.raw); }
    }

    /// Get a pointer to the underlying raw `fann_train_data` structure.
    pub unsafe fn get_raw(&self) -> *mut fann_sys::fann_train_data {
        self.raw
    }
}

impl Clone for TrainData {
    fn clone(&self) -> TrainData {
        unsafe {
            let raw = fann_sys::fann_duplicate_train_data(self.raw);
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
