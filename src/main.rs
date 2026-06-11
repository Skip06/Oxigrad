
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

impl Value {
    fn backward(&self) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
 
        Self::topo_sort(self, &mut topo, &mut visited); // means 1st all the children then the final node 

        // we will start applying the backward fn from the back 
        // derivative of final fn wrt itself is 1
        self.0.borrow_mut().grad = 1.0;
    
        for node in topo.iter().rev(){
            node.backward_grad();
        }
    }
}





impl Value {
    pub fn topo_sort(&self, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
        //But how to check if 2 nodes are same // CANT compare data or grad they might be sam3...
        // will check the pointer if they are present or not directly 
        let id = Rc::as_ptr(&self.0) as usize;
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);
        for child in self.0.borrow().prev.iter() {
            child.topo_sort(topo, visited);        //or we could have cloned it and do topo() with 3 arguments
        }
        topo.push(self.clone());
    }
}




fn main() {
    println!("Hello, world!");
}
