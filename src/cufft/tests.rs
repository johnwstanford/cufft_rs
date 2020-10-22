
use float_cmp::Ulps;
// use ieee754::Ieee754;

use rand::{thread_rng, Rng};

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use crate::runtime::cuda_vec::CudaVec;

use super::{PlanComplex1D, Complex64 as C64};

const NUM_ITERS:usize = 10;

// const ULP_THRESH_WARNING:i64 = 67108864;

// The significand for an f64 has 52 bits, so being off by 2^8 = 256 is still pretty insignificant; the most significant 44 bits still agree
const ULP_AVG_THRESH_FAIL_F64:f64 = 256.0;

#[test]
fn single_batch_fft() {

	let n:usize = 2usize.pow(16);
	let mut rng = thread_rng();

	let mut plan = PlanComplex1D::new(n as i32, 1).unwrap();

	let mut ulp_diff:Vec<f64> = vec![];

	for _ in 0..NUM_ITERS {
		let time_domain_host:Vec<C64> = (0..n).map(|_| C64(rng.gen_range(-100.0, 100.0), rng.gen_range(-100.0, 100.0))).collect();
		assert_eq!(time_domain_host.len(), n);	

		let time_domain_device = CudaVec::from_slice(&time_domain_host).unwrap();
		assert_eq!(time_domain_device.len(), n);

		let freq_domain_host = plan.fwd(&time_domain_device).unwrap();
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
			let cpu_re:f64 = freq_domain_cpu[i].re;
			let cpu_im:f64 = freq_domain_cpu[i].im;
			let gpu_re:f64 = freq_domain_host[i].0;
			let gpu_im:f64 = freq_domain_host[i].1;

			let ulps_re = cpu_re.ulps(&gpu_re).abs();
			let ulps_im = cpu_im.ulps(&gpu_im).abs();
			
			ulp_diff.push(ulps_re as f64);
			ulp_diff.push(ulps_im as f64);

			// if ulps_re > ULP_THRESH_WARNING {
			// 	eprintln!("WARNING: Real component at index {} off by {} ULPs: {:?} ({:?}) vs {:?} ({:?})", i, ulps_re, cpu_re, cpu_re.decompose_raw(), gpu_re, gpu_re.decompose_raw());
			// }

			// if ulps_im > ULP_THRESH_WARNING {
			// 	eprintln!("Imaginary component at index {} off by {} ULPs: {:?} ({:?}) vs {:?} ({:?})", i, ulps_im, cpu_im, cpu_im.decompose_raw(), gpu_im, gpu_im.decompose_raw());
			// }

		}

	}

	// This test needs to be statistical because there seem to always be outliers that are above any reasonable threshold
	let mu:f64 = ulp_diff.iter().sum::<f64>() / (ulp_diff.len() as f64);
	// let sigma_sq:f64 = ulp_diff.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / (ulp_diff.len() as f64);
	// eprintln!("ULP Difference = {:.3} +/- {:.3}", mu, sigma_sq.sqrt());
	assert_le!(mu, ULP_AVG_THRESH_FAIL_F64);

}