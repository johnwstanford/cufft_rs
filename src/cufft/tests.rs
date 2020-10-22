
use float_cmp::Ulps;
// use ieee754::Ieee754;

use rand::{thread_rng, Rng};

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use super::{Plan, ExecCudaFFT};

const NUM_ITERS:usize = 10;

// const ULP_THRESH_WARNING:i64 = 67108864;

// The significand for an f64 has 52 bits, so being off by 2^8 = 256 is still pretty insignificant; the most significant 44 bits still agree
const ULP_AVG_THRESH_FAIL_F64:f64 = 256.0;

#[test]
fn single_batch_fft_f32() {

	let n:usize = 2usize.pow(16);
	let mut rng = thread_rng();

	let mut plan = Plan::new(n as i32, 1).unwrap();

	let mut ulp_diff:Vec<f32> = vec![];

	for _ in 0..NUM_ITERS {
		let time_domain_host:Vec<Complex<f32>> = (0..n).map(|_| Complex{re: rng.gen_range(-100.0, 100.0), im: rng.gen_range(-100.0, 100.0)}).collect();
		assert_eq!(time_domain_host.len(), n);	

		let time_domain_slice:&[Complex<f32>] = &time_domain_host;

		let freq_domain_host:Vec<Complex<f32>> = time_domain_slice.fwd(&mut plan).unwrap();
		assert_eq!(freq_domain_host.len(), n);

		let freq_domain_cpu:Vec<Complex<f32>> = {
			let mut time_domain:Vec<Complex<f32>> = time_domain_host.clone();

			let mut freq_domain: Vec<Complex<f32>> = vec![Complex::zero(); n];
			let mut planner = FFTplanner::new(false);
			let fft = planner.plan_fft(n);

			// This function uses time_domain as scratch space, so it's to be considered garbage after this call
			fft.process(&mut time_domain, &mut freq_domain);

			freq_domain
		};

		// Compare the numerical values from the CPU and GPU
		for i in 0..n {
			let cpu_re:f32 = freq_domain_cpu[i].re;
			let cpu_im:f32 = freq_domain_cpu[i].im;
			let gpu_re:f32 = freq_domain_host[i].re;
			let gpu_im:f32 = freq_domain_host[i].im;

			let ulps_re = cpu_re.ulps(&gpu_re).abs();
			let ulps_im = cpu_im.ulps(&gpu_im).abs();
			
			ulp_diff.push(ulps_re as f32);
			ulp_diff.push(ulps_im as f32);

			// if ulps_re > ULP_THRESH_WARNING {
			// 	eprintln!("WARNING: Real component at index {} off by {} ULPs: {:?} ({:?}) vs {:?} ({:?})", i, ulps_re, cpu_re, cpu_re.decompose_raw(), gpu_re, gpu_re.decompose_raw());
			// }

			// if ulps_im > ULP_THRESH_WARNING {
			// 	eprintln!("Imaginary component at index {} off by {} ULPs: {:?} ({:?}) vs {:?} ({:?})", i, ulps_im, cpu_im, cpu_im.decompose_raw(), gpu_im, gpu_im.decompose_raw());
			// }

		}

	}

	// This test needs to be statistical because there seem to always be outliers that are above any reasonable threshold
	let mu:f32 = ulp_diff.iter().sum::<f32>() / (ulp_diff.len() as f32);
	assert_le!(mu, ULP_AVG_THRESH_FAIL_F64 as f32);

}

#[test]
fn single_batch_fft_f64() {

	let n:usize = 2usize.pow(16);
	let mut rng = thread_rng();

	let mut plan = Plan::new(n as i32, 1).unwrap();

	let mut ulp_diff:Vec<f64> = vec![];

	for _ in 0..NUM_ITERS {
		let time_domain_host:Vec<Complex<f64>> = (0..n).map(|_| Complex{re: rng.gen_range(-100.0, 100.0), im: rng.gen_range(-100.0, 100.0)}).collect();
		assert_eq!(time_domain_host.len(), n);	

		let time_domain_slice:&[Complex<f64>] = &time_domain_host;

		let freq_domain_host = time_domain_slice.fwd(&mut plan).unwrap();
		assert_eq!(freq_domain_host.len(), n);

		let freq_domain_cpu:Vec<Complex<f64>> = {
			let mut time_domain:Vec<Complex<f64>> = time_domain_host.clone();

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
			let gpu_re:f64 = freq_domain_host[i].re;
			let gpu_im:f64 = freq_domain_host[i].im;

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