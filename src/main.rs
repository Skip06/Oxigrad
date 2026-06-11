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

use std::{cell::RefCell, collections::HashSet, ops::{Add, Mul}, rc::Rc};

struct ValueData {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Value>,   
    pub op: String,
}

#[derive(Clone)]
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

    fn add(self, other: Self) -> Self{
        
        Value(Rc::new(RefCell::new(ValueData{
            data: self.0.borrow().data + other.0.borrow().data, //Value is tuple struct so inner val is accessed by idx .0
            grad: 0.0,
            prev: vec![ self.clone(), other.clone() ],  //.clone() does a deep copy with exclusive owership // but the thing u do Rc::clone() that should be wrapped with Rc<> and Value is not .
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
            prev: vec![self.clone(), rhs.clone()], 
            op: "*".to_string()
        })))
    }
}

impl Value{
    pub fn backward_grad(&self){
        let mut node = self.0.borrow_mut();  // node is now ValueData cause .0 gives tuples 1st elem which is  Rc<RefCell<ValueData>> and then .borrow() give the inner elem i.e. ValData type
        match node.op.as_str() {    //to convert String to &str => .as_str() cause match does not trigger deref coercion
            "+" => {                 // cant do = it as gradients must be accumulated. b = a + a example by karpathy // MULTIVARIATE DERIVATIVE
                node.prev[0].0.borrow_mut().grad += 1.0 * node.grad; // additon local derivative (1+0) * global derivative of final fn wrt this current node. 
                node.prev[1].0.borrow_mut().grad += 1.0 * node.grad;
            }
            "*" => { // just the chain rule of derivative 
                node.prev[0].0.borrow_mut().grad += node.prev[1].0.borrow().data * node.grad;
                node.prev[1].0.borrow_mut().grad += node.prev[0].0.borrow().data * node.grad;
               
            }

            _ => {}
            
        }       
    }
}




fn main() {
    println!("Hello, world!");
}
