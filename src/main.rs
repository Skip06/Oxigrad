use micrograd_rs::Value;

fn main() {
    let a = Value::new(3.0);
    let b = a.pow(2.0);  // b = a^2 = 9.0, db/da = 2*a = 6.0

    b.backward();

    println!("a grad: {}", a.grad());
    println!("b grad: {}", b.grad());

}