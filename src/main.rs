use formalith::{
    field::{VectorSpaceElement, M, R},
    parse, symbol,
};

pub fn main() {
    let test = parse!("x + 5 * y + (y+z) *3.55", R);
    println!("{}", test.expand());
    let a = parse!("1", R);
    let b = parse!("3", R);
    let c = parse!("6", R);
    let d = symbol!("x", R);
    let e = symbol!("y", R);
    let test = &d + &d;
    let test = &d + &test;
    println!("{}", test);
    let b = a + &b + &c + d + &e + &e;
    println!("{}", b);
    let b = (b.clone() + &b + b.clone() * &c) * parse!("10", R);
    println!("{} = {}", b, b.expand());
    println!("{}", parse!("(x*y)^2", R));
    // let a = parse!("(x+y)", R);
    // let mut s = a.clone();
    // for i in 0..18 {
    //     s = s * parse!(format!("x + y{}", i), R);
    // }
    // println!("{} = {}", s, s.expand().is_zero());
    // if b > c {

    // }
    // println!("{}", parse!("abs((x)^(1/4)^(2/4))", R).expand());
    // println!("{}", parse!("abs(abs(-3 + 2))", R));
    println!("{}", parse!("abs(3) * abs(3)", R));

    match parse!("[[2*(x^2)^3, 4], [2, 3]]", M) {
        formalith::term::Term::Value(value) => match value.get_value() {
            VectorSpaceElement::Vector(mut matrix) => {
                matrix.gaussian_elimination();
                println!("{}", matrix);
            }
            _ => {}
        },
        _ => {}
    }
    println!("{}", parse!("abs(3) *[[2*(x^2)^3, 4], [2, 3]]", M));
    // loop {
    //     let mut s = String::new();
    //     std::io::stdin().read_line(&mut s).unwrap();
    //     println!("{}", parse!(s, R));
    // }
}
