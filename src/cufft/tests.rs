
use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

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

	let freq_domain_cpu:Vec<Complex<f64>> = {
		let mut time_domain:Vec<Complex<f64>> = time_domain_host.iter().map(|x| Complex{ re: x.0, im: x.1 }).collect();

		let mut freq_domain: Vec<Complex<f64>> = vec![Complex::zero(); n];
		let mut planner = FFTplanner::new(false);
		let fft = planner.plan_fft(n);

		// This function uses time_domain as scratch space, so it's to be considered garbage after this call
		fft.process(&mut time_domain, &mut freq_domain);

		freq_domain
	};

	// Compare the numerical values from the CPU and GPU
	for i in 0..n {
		assert!( approx_eq!(f64, freq_domain_host[i].0, freq_domain_cpu[i].re, ulps = 2) );
		assert!( approx_eq!(f64, freq_domain_host[i].1, freq_domain_cpu[i].im, ulps = 2) );
	}


}