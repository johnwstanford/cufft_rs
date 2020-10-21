
use crate::runtime::cuda_vec::CudaVec;

use super::{PlanComplex1D, Complex64 as C64};

#[test]
fn single_batch_fft() {

	let n:usize = 4;

	let plan = PlanComplex1D::new(n as i32, 1).unwrap();

	let time_domain_host = vec![C64(1.0, 0.0), C64(2.2, 0.0), C64(3.0, 0.0), C64(4.0, 0.0)];
	assert_eq!(time_domain_host.len(), n);	

	let time_domain_device = CudaVec::from_slice(&time_domain_host).unwrap();
	assert_eq!(time_domain_device.len(), n);

	let freq_domain_device = plan.fwd(&time_domain_device).unwrap();
	assert_eq!(freq_domain_device.len(), n);

	let freq_domain_host   = freq_domain_device.clone_to_host().unwrap();
	assert_eq!(freq_domain_host.len(), n);

	// TODO: check the numerical value of the result
	// println!("{:?}", freq_domain_host);
}