
#[test]
fn basic_fft() {

	let plan = super::PlanComplex1D::new(16, 1).unwrap();

	println!("{:?}", plan);

}