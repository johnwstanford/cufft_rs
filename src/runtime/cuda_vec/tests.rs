
use super::CudaVec;

// const N:usize = 4;

#[test]
fn create_empty_vec() {

	let cuda_vec:CudaVec<f64> = CudaVec::new();

	println!("{:?}", cuda_vec);	
}