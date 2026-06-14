use crate::Value;
use rand::RngExt ;// bringing the RngExt trait

pub struct Neuron{
    pub w: Vec<Value>,
    pub b: Value,
    pub nonlin: bool,   // true = ReLU (hidden layers), false = linear (output layer)
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool) -> Self { // nin is number of i/p coming to the neuron
        let mut rng = rand::rng();
        let mut w = Vec::new();
        for _i in 0..nin{
            let weight = rng.random_range(-1.0..1.0);
            w.push(Value::new(weight));
        }
       return  Neuron { 
            w: w,
            b: Value::new(rng.random_range(-1.0..1.0)),
            nonlin,
        };
 
    }

    //  FINDS THE ACTIVATION OF A SINGLE NEURON 
    pub fn forward_neuron(&self, x: &[Value])-> Value{
        //w * x + b need to mult 2 arrays element wise => Zip 

        let mut activation = self.b.clone(); //making it owned type so out can be additve with w.clne() * x.clone()
        for (w, x) in self.w.iter().zip(x.iter()){
            activation = activation + w.clone() * x.clone(); // Mul requires owned types mut iter() returns immutable refs
        };

        if self.nonlin {
            activation.relu()   // hidden layers: apply ReLU
        } else {
            activation          // output layer: linear (no activation)
        }
    }

    pub fn parameters_neuron(&self) -> Vec<Value> {
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

pub struct Layer{
    pub neurons: Vec<Neuron>,
}

impl Layer{
    //nout is number of neurons in this layer.
    pub fn new(nout: usize, nin: usize, nonlin: bool) -> Self{  // every neuron of the same layer wiill have same number of inputs to it as they are weights

        let mut neurons = Vec::new();
        for _i in 0..nout{
            neurons.push(Neuron::new(nin, nonlin));
        }

        Self { 
            neurons: neurons,
        }
    }

    //NEED TO RETURN LIST OF ACTIVATION OF ALL NEURONS IN THIS LIST AS O/P SO IT CAN BE I/P TO THE NEXT LAYER//
    pub fn forward_layer(&self, x:&[Value]) -> Vec<Value>{

        self.neurons.iter().map(|neuron| neuron.forward_neuron(x)).collect()    // same x i/p to all neurons of this same layer 
    }  

    pub fn parameters_layer(&self) -> Vec<Value>{
        self.neurons.iter().flat_map(|n| n.parameters_neuron()).collect()
        
    }
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP{
    pub fn new( nouts: &[usize],nin: usize,) -> Self { // nin is starting ip size and nout is arr of size of everty layer uske baad  

            let mut sizes = vec![nin];
            sizes.extend_from_slice(nouts);// combine the input size and output sizes into one list to build layers
    
            let mut layers = Vec::new();
            let last = nouts.len() - 1;
            for i in 0..nouts.len(){
                let nonlin = i != last;  // hidden layers = true (ReLU), output layer = false (linear)
                layers.push(Layer::new(sizes[i+1], sizes[i], nonlin));
            }
            
            Self {
                layers, 
            }
    }
    
    pub fn forward_pass(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec(); //x -> layer1.forward_pass(x) -> layer2.forward_pass(layer1_output)

    
        for layer in self.layers.iter() {
            out = layer.forward_layer(&out);
        }
    
        out
    }

    pub fn parameters(&self)->Vec<Value>{
        self.layers.iter().flat_map(|l| l.parameters_layer()).collect()
    }

    
}