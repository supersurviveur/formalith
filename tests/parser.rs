use formalith::{
    field::{
        Set,
        matrix::{M, VectorSpaceElement},
        real::R,
    },
    matrix::Matrix,
    parse, try_parse,
};
use malachite::Rational;

#[test]
fn test_parse() {
    assert_eq!(parse!("5", R), 5.into());
    assert_eq!(
        parse!("5.5^2", R),
        Rational::try_from_float_simplest(30.25).unwrap().into()
    );
    assert_eq!(
        parse!("[[5]]", M),
        VectorSpaceElement::Vector(Matrix::new((1, 1), vec![5.into()], R.get_term_set())).into()
    );
}
#[test]
fn test_try_parse() {
    assert!(try_parse!("5+", R).is_err());
    assert_eq!(try_parse!("-+-+--5", R), Ok(5.into()));
    assert!(try_parse!("[5]", R).is_err());
    // Can't recurse more than 10 times inside upper set
    assert!(try_parse!("det(det(det(det(det(det(det(det(det(det(5))))))))))", R).is_err());
}
