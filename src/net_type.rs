use fann_sys::*;

/// Network types
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum NetType {
    /// Each layer of neurons only has connections to the next layer.
    Layer,
    /// Each layer has connections to all following layers.
    Shortcut,
}

impl NetType {
    /// Create a `NetType` from a `fann_sys::fann_nettype_enum`.
    pub fn from_nettype_enum(nt_enum: fann_nettype_enum) -> NetType {
        match nt_enum {
            FANN_NETTYPE_LAYER => NetType::Layer,
            FANN_NETTYPE_SHORTCUT => NetType::Shortcut,
        }
    }
}
