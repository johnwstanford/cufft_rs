
use super::CudaVec;

// const N:usize = 4;

#[test]
fn create_and_resize_vec() {

	let mut cuda_vec:CudaVec<f64> = CudaVec::new();

	println!("{:?}", cuda_vec);	

	cuda_vec.set_capacity(10).unwrap();

	println!("{:?}", cuda_vec);	
	
	cuda_vec.set_capacity(5).unwrap();

	println!("{:?}", cuda_vec);	
	
}