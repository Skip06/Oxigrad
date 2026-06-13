
use std::{cell::RefCell, collections::HashSet, ops::{Add, Mul}, rc::Rc};

struct ValueData {
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Value>,
    pub op: String,
    pub exp: f64,
}

#[derive(Clone)]
struct Value(Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f64) -> Self { // its like doing => let a = Value::new(4.0);
        let out = Value (Rc::new(RefCell::new(ValueData{
            data,
            grad: 0.0,
            prev: vec![],
            op: String::new(),
            exp: 0.0,
        })));
        return out ;
    }
}

impl Add for Value{   // we will be returning a Value type after adding with its properties
    type Output = Value;

    fn add(self, other: Self) -> Self{
        
        let out = Value(Rc::new(RefCell::new(ValueData{
            data: self.0.borrow().data + other.0.borrow().data, //Value is tuple struct so inner val is accessed by idx .0
            grad: 0.0,
            prev: vec![ self.clone(), other.clone() ],  //.clone() does a deep copy with exclusive owership // but the thing u do Rc::clone() that should be wrapped with Rc<> and Value is not .
            op: "+".to_string(),
            exp: 0.0,
        })));
        return out ; 
    }
}

impl Mul for Value{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
       let out = Value(Rc::new(RefCell::new(ValueData{
            data: self.0.borrow().data * rhs.0.borrow().data, 
            grad: 0.0,
            prev: vec![self.clone(), rhs.clone()], 
            op: "*".to_string(),
            exp: 0.0,
        })));
       return out ;
    }
}

impl Value {
    pub fn pow(&self, exp: f64) -> Self {
        let out = Value(Rc::new(RefCell::new(ValueData {
            data: self.0.borrow().data.powf(exp),
            grad: 0.0,
            prev: vec![self.clone()],
            op: "pow".to_string(),
            exp,
        })));
        return out;
    }
}

impl Value{
    pub fn relu(&self) -> Self{
        let out = Value(Rc::new(RefCell::new(ValueData{
            data: if self.0.borrow().data < 0.0 {0.0} else {self.0.borrow().data },
            grad: 0.0,
            prev: vec![self.clone()],
            op: "ReLU".to_string(),
            exp: 0.0,
        })));
        return out; 
    }
   
}


impl Value{
    pub fn backward_grad(&self){
        let node = self.0.borrow_mut();  // node is now ValueData cause .0 gives tuples 1st elem which is  Rc<RefCell<ValueData>> and then .borrow() give the inner elem i.e. a Ref to ValueData but writing .grad autoDeref it to access the value like *
        match node.op.as_str() {    //to convert String to &str => .as_str() cause match does not trigger deref coercion
            "+" => {                 // cant do = it as gradients must be accumulated. b = a + a example by karpathy // MULTIVARIATE DERIVATIVE
                node.prev[0].0.borrow_mut().grad += 1.0 * node.grad; // additon local derivative (1+0) * global derivative of final fn wrt this current node. 
                node.prev[1].0.borrow_mut().grad += 1.0 * node.grad;
            }
            "*" => { // just the chain rule of derivative 
                node.prev[0].0.borrow_mut().grad += node.prev[1].0.borrow().data * node.grad;
                node.prev[1].0.borrow_mut().grad += node.prev[0].0.borrow().data * node.grad;
                
            }
            "pow" => {
                let exp = node.exp;
                let base = node.prev[0].0.borrow().data;
                node.prev[0].0.borrow_mut().grad += exp * base.powf(exp - 1.0) * node.grad;
            }
            "ReLU" => {
                if node.data > 0.0 {
                    node.prev[0].0.borrow_mut().grad += 1.0 * node.grad;
                }
                else{
                    node.prev[0].0.borrow_mut().grad += 0.0;
                }
            }

            _ => {}
            
        }       
    }

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
    let a = Value::new(3.0);
    let b = a.pow(2.0);  // b = a^2 = 9.0, db/da = 2*a = 6.0

    b.backward();

    println!("a grad: {}", a.0.borrow().grad); // Should be 6.0
    println!("b grad: {}", b.0.borrow().grad); // Should be 1.0
}