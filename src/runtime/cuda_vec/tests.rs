
use crate::runtime::{PointerAttributes, MemoryType};

use super::CudaVec;

// const N:usize = 4;

#[test]
fn create_and_resize_vec() {

	let mut cuda_vec:CudaVec<f64> = CudaVec::new();

	assert_eq!(cuda_vec.len(),      0);
	assert_eq!(cuda_vec.capacity(), 0);
	assert!(PointerAttributes::get(cuda_vec.as_ptr() as *const u8).is_err());

	cuda_vec.set_capacity(10).unwrap();
	assert_eq!(cuda_vec.len(),      0 );
	assert_eq!(cuda_vec.capacity(), 10);
	assert!(PointerAttributes::get(cuda_vec.as_ptr() as *const u8).unwrap().typ == MemoryType::Device);
	let ptr0 = cuda_vec.as_ptr();

	cuda_vec.set_capacity(5).unwrap();
	assert_eq!(cuda_vec.len(),      0);
	assert_eq!(cuda_vec.capacity(), 5);
	assert!(PointerAttributes::get(cuda_vec.as_ptr() as *const u8).unwrap().typ == MemoryType::Device);
	let ptr1 = cuda_vec.as_ptr();

	assert!(ptr0 != ptr1);

	let host_vec0:Vec<f64> = vec![1.1, 2.2, 0.0, 3.3];
	cuda_vec.copy_from_slice(&host_vec0).unwrap();
	assert_eq!(cuda_vec.len(),      host_vec0.len());
	assert_eq!(cuda_vec.capacity(), host_vec0.len());

	let host_vec1:Vec<f64> = cuda_vec.clone_to_host().unwrap();

	assert_eq!(host_vec0, host_vec1);

}