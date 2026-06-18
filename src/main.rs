use micrograd_rs::{Value, micrograd::nn::MLP, graphiz};


fn main() {
    // this below is a dummy dataset from karpathy's video
    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let ys = vec![
        Value::new(1.0),
        Value::new(-1.0),
        Value::new(-1.0),
        Value::new(1.0),
    ];

    let mlp = MLP::new(&[4, 4, 1], 3); // nn with 3 imputs , two hiddenlayes with 4 neurons, 1 output
    let epochs = 30;
    let lr = 0.04;

    let mut loss = Value::new(0.0); // hoisted so draw_dot can access it after the loop

    for epoch in 0..epochs {
        let mut y_nn = Vec::new();
        for x in &xs {
            let pred = mlp.forward_pass(x)[0].clone(); // we jsut have one output neuron
            y_nn.push(pred);
        }

        loss = Value::new(0.0);
        for (y_pred, y) in y_nn.iter().zip(ys.iter()) {
            let diff = y_pred.clone() - y.clone(); //jsut finding the loss
            let diff_sq = diff.clone() * diff.clone();
            loss = loss + diff_sq;
        }

        //ZERO GRADIENT
        for param in mlp.parameters() {
            param.zero_grad();
        }

        loss.backward();

        for param in mlp.parameters() {
            //gonna update every single parameter
            let grad = param.grad();
            //param.data() -= lr * grad; ERROR as we need something to into RefCell and change this fucntion call is not actually changin that value
            let update = param.data() - lr * grad;
            param.set_data(update);
        }

        println!("Epoch {}: Loss = {}", epoch, loss.data());
    }

    // ── Visualize the compute graph ──
    graphiz::render(&loss, "loss", "graph");
}
