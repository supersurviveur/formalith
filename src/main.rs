use formalith::{
    field::{TryElementCast, M, R, Z},
    parse,
    polynom::MultivariatePolynomial,
    symbol,
};
use malachite::{rational::Rational, Integer};

pub fn main() {
    println!("{}", parse!("(x+y)^4", R).expand());
    println!("{}", parse!("71/1200 + (x^6*2)^-1", R));

    println!("{}", parse!("[[x, 4], [2, 3]]^-1", M).simplify());
    println!("{}", parse!("[[x, 4], [2, 3]]^-1", M).expand());
    println!("{}", parse!("[[x, 4], [2, 3]]^-1", M).expand());

    // match parse!("[[x, 4], [2, 3]]", M) {
    //     formalith::term::Term::Value(value) => match value.get_value() {
    //         VectorSpaceElement::Vector(matrix) => {
    //             // matrix.partial_row_reduce();
    //             println!("{}\n=\n{}", matrix, matrix.inv().unwrap());
    //             println!("{}", matrix.det().unwrap());
    //         }
    //         _ => {}
    //     },
    //     _ => {}
    // }
    // println!(
    //     "{}",
    //     parse!("[[1, 4, 5], [2, 3^x, 6], [2, 3, 6]]^-1", M)
    // );
    // println!(
    //     "{}",
    //     parse!("(((3^x-8)^-1)*(-4*(3^x-8)-20))^-1", R).expand()
    // );
    // println!("{}", parse!("abs(3) *[[2*(x^2)^3, 4], [2, 3]]", M));
    // loop {
    //     let mut s = String::new();
    //     std::io::stdin().read_line(&mut s).unwrap();
    //     println!("{}", parse!(s, R));
    // }
    // let mut comb_iter = CompositionIterator::new(3, 2);
    // while let Some(combination) = comb_iter.next() {
    //     println!("{:?}", combination);
    // }
    //
    let p = MultivariatePolynomial::new(
        vec![
            (vec![(symbol!("x"), Rational::from(3))], Rational::from(1)),
            (vec![(symbol!("x"), Rational::from(2))], Rational::from(3)),
            (vec![(symbol!("x"), Rational::from(1))], Rational::from(3)),
            (vec![], Rational::from(1)),
        ],
        R,
        R,
    );
    // println!(
    //     "{}",
    //     p.gcd(&MultivariatePolynomial::new(
    //         vec![
    //             (vec![(symbol!("x"), Rational::from(1))], Rational::from(2)),
    //             (vec![], Rational::from(1))
    //         ],
    //         R,
    //         R
    //     ))
    // );
    println!("{}", parse!("(x + 3)^3", R).to_polynomial());
    println!("=");
    println!("{}", parse!("(x + 3)^3", R).expand().factor());
    // println!(
    //     "{} = ({})^{}",
    //     p,
    //     p.square_free_factorization()[0].0,
    //     p.square_free_factorization()[0].1
    // );
    let matrix = parse!("[[x, 4], [2, 3]]", M);
    let det = matrix.det().unwrap();
    println!("{}", det);
    let det = parse!("det([[x, 4], [2, 3]])", R);
    println!("{}", det);
}
