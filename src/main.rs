use micrograd_rs::{Value, micrograd::nn::MLP};

fn main() {

    // this below is a dummy dataset from karpathy's video 
    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let ys = vec![Value::new(1.0),Value::new(-1.0),Value::new(-1.0),Value::new(1.0)];

    let mlp = MLP::new(&[4,4,1], 3); // nn with 3 imputs , two hiddenlayes with 4 neurons, 1 output
    let epochs = 20;
    let lr = 0.04;

    for epoch in 0..epochs{
        let mut y_nn = Vec::new();
        for x in &xs{
            let pred = mlp.forward_pass(x)[0].clone();  // we jsut have one output neuron
            y_nn.push(pred);
        }

        let mut loss = Value::new(0.0);
        for (y_pred, y) in y_nn.iter().zip(ys.iter()){
            let diff = y_pred.clone() - y.clone();         //jsut finding the loss
            let diff_sq = diff.clone() * diff.clone();
            loss = loss + diff_sq;
        }

        //ZERO GRADIENT
        for param in mlp.parameters(){
            param.zero_grad();
        }

        loss.backward();

        for param in mlp.parameters(){
            let grad = param.grad();
            let updated = param.data() - lr * grad;
            param.set_data(updated);
        }
        

        
    }
    
}