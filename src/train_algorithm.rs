use libc::c_float;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct IncrementalParams {
    /// A higher momentum can be used to speed up incremental training. It should be between 0
    /// and 1, the default is 0.
    pub learning_momentum: c_float,
    /// The learning rate determines how aggressive training should be. Default is 0.7.
    pub learning_rate: c_float,
}

impl Default for IncrementalParams {
    fn default() -> IncrementalParams {
        IncrementalParams {
            learning_momentum: 0.0,
            learning_rate: 0.7,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct BatchParams {
    /// The learning rate determines how aggressive training should be. Default is 0.7.
    pub learning_rate: c_float,
}

impl Default for BatchParams {
    fn default() -> BatchParams {
        BatchParams { learning_rate: 0.7 }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RpropParams {
    /// A value less than 1, used to decrease the step size during training. Default 0.5
    pub decrease_factor: c_float,
    /// A value greater than 1, used to increase the step size during training. Default 1.2
    pub increase_factor: c_float,
    /// The minimum step size. Default 0.0
    pub delta_min: c_float,
    /// The maximum step size. Default 50.0
    pub delta_max: c_float,
    /// The initial step size. Default 0.1
    pub delta_zero: c_float,
}

impl Default for RpropParams {
    fn default() -> RpropParams {
        RpropParams {
            decrease_factor: 0.5,
            increase_factor: 1.2,
            delta_min: 0.0,
            delta_max: 50.0,
            delta_zero: 0.1,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct QuickpropParams {
    /// The factor by which weights should become smaller in each iteration, to ensure that
    /// the weights don't grow too large during training. Should be a negative number close to
    /// 0. The default is -0.0001.
    pub decay: c_float,
    /// The mu factor is used to increase or decrease the step size; should always be greater
    /// than 1. The default is 1.75.
    pub mu: c_float,
    /// The learning rate determines how aggressive training should be. Default is 0.7.
    pub learning_rate: c_float,
}

impl Default for QuickpropParams {
    fn default() -> QuickpropParams {
        QuickpropParams {
            decay: -0.0001,
            mu: 1.75,
            learning_rate: 0.7,
        }
    }
}

/// The Training algorithms used when training on `fann_train_data` with functions like
/// `fann_train_on_data` or `fann_train_on_file`. The incremental training alters the weights
/// after each time it is presented an input pattern, while batch only alters the weights once after
/// it has been presented to all the patterns.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum TrainAlgorithm {
    /// Standard backpropagation algorithm, where the weights are updated after each training
    /// pattern. This means that the weights are updated many times during a single epoch and some
    /// problems will train very fast, while other more advanced problems will not train very well.
    Incremental(IncrementalParams),
    /// Standard backpropagation algorithm, where the weights are updated after calculating the mean
    /// square error for the whole training set. This means that the weights are only updated once
    /// during an epoch. For this reason some problems will train slower with this algorithm. But
    /// since the mean square error is calculated more correctly than in incremental training, some
    /// problems will reach better solutions.
    Batch(BatchParams),
    /// A more advanced batch training algorithm which achieves good results for many problems.
    /// `Rprop` is adaptive and therefore does not use the `learning_rate`. Some other parameters
    /// can, however, be set to change the way `Rprop` works, but it is only recommended for users
    /// with a deep understanding of the algorithm. The original RPROP training algorithm is
    /// described by [Riedmiller and Braun, 1993], but the algorithm used here is a variant, iRPROP,
    /// described by [Igel and Husken, 2000].
    Rprop(RpropParams),
    /// A more advanced batch training algorithm which achieves good results for many problems. The
    /// quickprop training algorithm uses the `learning_rate` parameter along with other more
    /// advanced parameters, but it is only recommended to change these for users with a deep
    /// understanding of the algorithm. Quickprop is described by [Fahlman, 1988].
    Quickprop(QuickpropParams),
}

impl Default for TrainAlgorithm {
    fn default() -> TrainAlgorithm {
        TrainAlgorithm::Rprop(Default::default())
    }
}
