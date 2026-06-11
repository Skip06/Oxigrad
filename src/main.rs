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

use std::{cell::RefCell, ops::{Add, Mul}, rc::Rc};

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

impl Add for Value{   // we will be returning a Value type after adding with its properties
    type Output = Value;

    fn add(self, other: Value) -> Self{
        
        Value(Rc::new(RefCell::new(ValueData{
            data: self.0.borrow().data + other.0.borrow().data, //Value is tuple struct so inner val is accessed by idx .0
            grad: 0.0,
            prev: vec![Rc::clone(&self.0), Rc::clone(&other.0)],  //.clone() does a deep copy with exclusive owership
            op: "+".to_string()
        })))
        
    }
}

impl Mul for Value{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Value(Rc::new(RefCell::new(ValueData{
            data: self.0.borrow().data * rhs.0.borrow().data, 
            grad: 0.0,
            prev: vec![Rc::clone(&self.0), Rc::clone(&rhs.0)], 
            op: "*".to_string()
        })))
    }
}




fn main() {
    println!("Hello, world!");
}
