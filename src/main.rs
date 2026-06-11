//This thing wont work cause we need to do operations on our Values
//and if i Rc<RefCell<Value>> then cant impl add or mul trait cause of orphan rule .
// also too much to write cause manual .borrow() and clone(). 
// SO WE NEED A WRAPPER. => tuple struct 

// struct Value {
//     pub data: f64,
//     pub grad: f64,
//     // We have to wrap the parents so multiple nodes can point to them
//     pub prev: Vec<Rc<RefCell<Value>>>, 
//     pub op: String, 
// }

use std::{cell::RefCell, rc::Rc};

struct ValueData {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Rc<RefCell<ValueData>>>,
    pub op: String,
}

struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Self { // its like doing => let a = Value::new(4.0);
        Value (Rc::new(RefCell::new(ValueData{
            data,
            grad: 0.0,
            prev: vec![],
            op: String::new(),
        })))
    }
}





fn main() {
    println!("Hello, world!");
}
