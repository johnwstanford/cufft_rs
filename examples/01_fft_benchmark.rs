
use std::time::Instant;

use rand::{thread_rng, Rng};

use rustfft::FFTplanner;
use rustfft::num_complex::Complex;
use rustfft::num_traits::Zero;

use cufft_rs::cufft::{Plan, ExecCudaFFT};

fn main() -> Result<(), &'static str> {

	let n:usize = 2usize.pow(26);
	println!("FFT size = {:.3e}", n);

	let mut rng = thread_rng();

	let mut plan = Plan::new(n as i32, 1).unwrap();

	let time_domain_host:Vec<Complex<f32>> = (0..n).map(|_| Complex{re: rng.gen_range(-100.0, 100.0), im: rng.gen_range(-100.0, 100.0)}).collect();

	let time_domain_slice:&[Complex<f32>] = &time_domain_host;

	let gpu_start = Instant::now();
	let freq_domain_host:Vec<Complex<f32>> = time_domain_slice.fwd(&mut plan).unwrap();
	println!("GPU time: {:.4} [sec]", gpu_start.elapsed().as_secs_f32());

	let freq_domain_cpu:Vec<Complex<f32>> = {
		let mut time_domain:Vec<Complex<f32>> = time_domain_host.clone();

		let mut freq_domain: Vec<Complex<f32>> = vec![Complex::zero(); n];
		let mut planner = FFTplanner::new(false);
		let fft = planner.plan_fft(n);

		// This function uses time_domain as scratch space, so it's to be considered garbage after this call
		let cpu_start = Instant::now();
		fft.process(&mut time_domain, &mut freq_domain);
		println!("CPU time: {:.4} [sec]", cpu_start.elapsed().as_secs_f32());

		freq_domain
	};

	println!("First few freq domain values:");
	for i in 0..5 {
		println!("{} CPU: {:25.3}, GPU: {:25.3}", i, freq_domain_cpu[i], freq_domain_host[i]);
	}

	Ok(())

}