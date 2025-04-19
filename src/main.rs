use formalith::{
    field::{VectorSpaceElement, M, R},
    parse,
};

pub fn main() {
    println!("{}", parse!("71/1200 + (x^6*2)^-1", R));
    println!("{}", parse!("abs((x*y)^2)", R));
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

    // match parse!("[[2*(x^2)^3, 4], [2, 3]]", M) {
    match parse!("[[1, 4], [2, 3]]", M) {
        formalith::term::Term::Value(value) => match value.get_value() {
            VectorSpaceElement::Vector(matrix) => {
                // matrix.partial_row_reduce();
                println!("{}\n=\n{}", matrix, matrix.inv().unwrap());
                println!("{}", matrix.det().unwrap());
            }
            _ => {}
        },
        _ => {}
    }
    println!(
        "{}",
        parse!("abs(x) *[[1, 4, 5], [2, 3^x, 6], [2, 3, 6]]", M)
    );
    // println!("{}", parse!("abs(3) *[[2*(x^2)^3, 4], [2, 3]]", M));
    // loop {
    //     let mut s = String::new();
    //     std::io::stdin().read_line(&mut s).unwrap();
    //     println!("{}", parse!(s, R));
    // }
}
