use crate::Value;
use rand::RngExt ;// bringing the RngExt trait

pub struct Neuron{
    pub w: Vec<Value>,
    pub b: Value,
}


