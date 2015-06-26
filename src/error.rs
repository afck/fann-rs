use fann_sys::{fann_error, fann_get_errno, fann_get_errstr};
use fann_sys::fann_errno_enum::*;
use self::FannErrorType::*;
use std::error::Error;
use std::fmt;
use std::ffi::CStr;

#[derive(Copy, Clone, Debug)]
pub enum FannErrorType {
    /// Unable to open configuration file for reading
    CantOpenConfigR,
    /// Unable to open configuration file for writing
    CantOpenConfigW,
    /// Wrong version of configuration file
    WrongConfigVersion,
    /// Error reading info from configuration file
    CantReadConfig,
    /// Error reading neuron info from configuration file
    CantReadNeuron,
    /// Error reading connections from configuration file
    CantReadConnections,
    /// Number of connections not equal to the number expected
    WrongNumConnections,
    /// Unable to open train data file for writing
    CantOpenTdW,
    /// Unable to open train data file for reading
    CantOpenTdR,
    /// Error reading training data from file
    CantReadTd,
    /// Unable to allocate memory
    CantAllocateMem,
    /// Unable to train with the selected activation function
    CantTrainActivation,
    /// Unable to use the selected activation function
    CantUseActivation,
    /// Irreconcilable differences between two `fann_train_data` structures
    TrainDataMismatch,
    /// Unable to use the selected training algorithm
    CantUseTrainAlg,
    /// Trying to take subset which is not within the training set
    TrainDataSubset,
    /// Index is out of bound
    IndexOutOfBound,
    /// Scaling parameters not present
    ScaleNotPresent,
    // Errors specific to the Rust wrapper:
    /// Failed to save file
    CantSaveFile,
}

impl fmt::Display for FannErrorType {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        fmt::Display::fmt(match *self {
            CantOpenConfigR     => "Unable to open configuration file for reading",
            CantOpenConfigW     => "Unable to open configuration file for writing",
            WrongConfigVersion  => "Wrong version of configuration file",
            CantReadConfig      => "Error reading info from configuration file",
            CantReadNeuron      => "Error reading neuron info from configuration file",
            CantReadConnections => "Error reading connections from configuration file",
            WrongNumConnections => "Number of connections not equal to the number expected",
            CantOpenTdW         => "Unable to open train data file for writing",
            CantOpenTdR         => "Unable to open train data file for reading",
            CantReadTd          => "Error reading training data from file",
            CantAllocateMem     => "Unable to allocate memory",
            CantTrainActivation => "Unable to train with the selected activation function",
            CantUseActivation   => "Unable to use the selected activation function",
            TrainDataMismatch   => "Irreconcilable differences between two Fann objects",
            CantUseTrainAlg     => "Unable to use the selected training algorithm",
            TrainDataSubset     => "Trying to take subset which is not within the training set",
            IndexOutOfBound     => "Index is out of bound",
            ScaleNotPresent     => "Scaling parameters not present",
            CantSaveFile        => "Failed saving file",
        }, f)
    }
}

#[derive(Clone, Debug)]
pub struct FannError {
    pub error_type: FannErrorType,
    pub error_str: String,
}

impl fmt::Display for FannError {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        try!(self.error_type.fmt(f));
        try!(": ".fmt(f));
        self.error_str.fmt(f)
    }
}

impl Error for FannError {
    fn description(&self) -> &str {
        &self.error_str[..]
    }
}

impl FannError {
    /// Returns an `Err` if the previous operation on `errdat` has resulted in an error, otherwise
    /// `Ok(())`.
    pub unsafe fn check_no_error(errdat: *mut fann_error) -> Result<(), FannError> {
        if errdat.is_null() {
            return Err(FannError {
                error_type: FannErrorType::CantAllocateMem,
                error_str: "Unable to create a new object".to_string(),
            });
        }
        let error_type = match fann_get_errno(errdat) {
            FANN_E_NO_ERROR              => return Ok(()),
            FANN_E_CANT_OPEN_CONFIG_R    => CantOpenConfigR,
            FANN_E_CANT_OPEN_CONFIG_W    => CantOpenConfigW,
            FANN_E_WRONG_CONFIG_VERSION  => WrongConfigVersion,
            FANN_E_CANT_READ_CONFIG      => CantReadConfig,
            FANN_E_CANT_READ_NEURON      => CantReadNeuron,
            FANN_E_CANT_READ_CONNECTIONS => CantReadConnections,
            FANN_E_WRONG_NUM_CONNECTIONS => WrongNumConnections,
            FANN_E_CANT_OPEN_TD_W        => CantOpenTdW,
            FANN_E_CANT_OPEN_TD_R        => CantOpenTdR,
            FANN_E_CANT_READ_TD          => CantReadTd,
            FANN_E_CANT_ALLOCATE_MEM     => CantAllocateMem,
            FANN_E_CANT_TRAIN_ACTIVATION => CantTrainActivation,
            FANN_E_CANT_USE_ACTIVATION   => CantUseActivation,
            FANN_E_TRAIN_DATA_MISMATCH   => TrainDataMismatch,
            FANN_E_CANT_USE_TRAIN_ALG    => CantUseTrainAlg,
            FANN_E_TRAIN_DATA_SUBSET     => TrainDataSubset,
            FANN_E_INDEX_OUT_OF_BOUND    => IndexOutOfBound,
            FANN_E_SCALE_NOT_PRESENT     => ScaleNotPresent,
        };
        let errstr_bytes = CStr::from_ptr(fann_get_errstr(errdat)).to_bytes().to_vec();
        let error_str = String::from_utf8(errstr_bytes)
            .unwrap_or("Invalid UTF-8 in error string".to_string());
        Err(FannError {
            error_type: error_type,
            error_str: error_str,
        })
    }
}

