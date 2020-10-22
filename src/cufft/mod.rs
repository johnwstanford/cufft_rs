
use libc::size_t;

use rustfft::num_complex::Complex;

use crate::CError;
use crate::runtime::cuda_vec::CudaVec;

#[cfg(test)]
mod tests;

#[link(name = "cufft")]
extern {

    fn cufftPlanMany(plan:&mut size_t, rank:i32, n:&i32, inembed:&i32, istride:i32, idist:i32,
        onembed:&i32, ostride:i32, odist:i32, fft_type:cufftType_t, batch:i32) -> cufftResult_t;

	fn cufftExecZ2Z(plan:size_t, idata:*const Complex<f64>, odata:*mut Complex<f64>, direction:Direction) -> CError;
	fn cufftExecC2C(plan:size_t, idata:*const Complex<f32>, odata:*mut Complex<f32>, direction:Direction) -> CError;

    fn cufftDestroy(plan:size_t) -> cufftResult_t;

}

#[derive(Debug)]
pub struct Plan<T: Clone + Default, U: Clone + Default> {
    handle:usize,
    n:i32,          // --- Size of the Fourier transform
    batch_count:i32,
    in_buffer:CudaVec<T>,
    out_buffer:CudaVec<U>,
}

impl<T: Clone + Default, U: Clone + Default> Plan<T, U> {

    pub fn new(n:i32, batch_count:i32) -> Result<Self, &'static str> {
        if batch_count != 1 { return Err("TODO: implement batch counts other than 1"); }

        let handle:usize = 0;

    	let in_buffer  = CudaVec::from_slice(&vec![T::default(); n as usize])?;
    	let out_buffer = CudaVec::from_slice(&vec![U::default(); n as usize])?;

		Ok(Plan{ handle, n, batch_count, in_buffer, out_buffer })

    }

    pub fn is_initialized(&self) -> bool { self.handle != 0 }

}

impl<T: Clone + Default, U: Clone + Default> std::ops::Drop for Plan<T, U> {

    fn drop(&mut self) {
    	if self.is_initialized() {
	        unsafe { cufftDestroy(self.handle); }
    	}
    }

}

// TODO: make these more advanced features available
const RANK:i32 = 1; 				// --- 1D FFTs
const I_STRIDE:i32 = 1;				// --- Distance between two successive input/output elements
const O_STRIDE:i32 = 1;
const I_N_EMBED:[i32; 1] = [0];		// --- Input size with pitch (ignored for 1D transforms)
const O_N_EMBED:[i32; 1] = [0];     // --- Output size with pitch (ignored for 1D transforms)

trait InitCudaFFT {

	fn init(&mut self) -> Result<(), &'static str>;

}

impl InitCudaFFT for Plan<Complex<f64>, Complex<f64>> {
	
	fn init(&mut self) -> Result<(), &'static str> {

		if self.is_initialized() { return Err("Plan already initialized"); }
        
        match unsafe { cufftPlanMany(&mut self.handle, RANK, &self.n, 
                &I_N_EMBED[0], I_STRIDE, self.n,
                &O_N_EMBED[0], O_STRIDE, self.n, 
                cufftType_t::Z2Z, self.batch_count) } {
            cufftResult_t::Success => Ok(()),
            _                      => Err("Unable to initialize FFT plan")
        }

	}

}

impl InitCudaFFT for Plan<Complex<f32>, Complex<f32>> {
	
	fn init(&mut self) -> Result<(), &'static str> {

		if self.is_initialized() { return Err("Plan already initialized"); }
        
        match unsafe { cufftPlanMany(&mut self.handle, RANK, &self.n, 
                &I_N_EMBED[0], I_STRIDE, self.n,
                &O_N_EMBED[0], O_STRIDE, self.n, 
                cufftType_t::C2C, self.batch_count) } {
            cufftResult_t::Success => Ok(()),
            _                      => Err("Unable to initialize FFT plan")
        }

	}

}

pub trait ExecCudaFFT<T: Clone + Default, U: Clone + Default> {

	// Take an immutable reference to self, then use the given plan to execute a forward FFT
	fn fwd(&self, plan:&mut Plan<T, U>) -> Result<Vec<U>, &'static str>;

}

impl ExecCudaFFT<Complex<f64>, Complex<f64>> for &[Complex<f64>] {

	fn fwd(&self, plan:&mut Plan<Complex<f64>, Complex<f64>>) -> Result<Vec<Complex<f64>>, &'static str> {

		if !plan.is_initialized() { plan.init()?; }

    	if self.len() != plan.n as usize { 
    		Err("Wrong sized input") 
    	} else {
			plan.in_buffer.copy_from_slice(&self)?;
	    	unsafe { cufftExecZ2Z(plan.handle, plan.in_buffer.as_ptr() as *const _, plan.out_buffer.as_mut_ptr() as *mut _, Direction::Forward).ok()?; }
	    	Ok(plan.out_buffer.clone_to_host()?)		
    	}
	}

}

impl ExecCudaFFT<Complex<f32>, Complex<f32>> for &[Complex<f32>] {

	fn fwd(&self, plan:&mut Plan<Complex<f32>, Complex<f32>>) -> Result<Vec<Complex<f32>>, &'static str> {

		if !plan.is_initialized() { plan.init()?; }

    	if self.len() != plan.n as usize { 
    		Err("Wrong sized input") 
    	} else {
			plan.in_buffer.copy_from_slice(&self)?;
	    	unsafe { cufftExecC2C(plan.handle, plan.in_buffer.as_ptr() as *const _, plan.out_buffer.as_mut_ptr() as *mut _, Direction::Forward).ok()?; }
	    	Ok(plan.out_buffer.clone_to_host()?)		
    	}
	}

}

#[repr(C)]
pub enum Direction {
	Forward = -1,
	Inverse =  1,
}

#[repr(C)]
pub enum cufftType_t {
  R2C = 0x2a,     // Real to Complex (interleaved)
  C2R = 0x2c,     // Complex (interleaved) to Real
  C2C = 0x29,     // Complex to Complex, interleaved
  D2Z = 0x6a,     // Double to Double-Complex
  Z2D = 0x6c,     // Double-Complex to Double
  Z2Z = 0x69      // Double-Complex to Double-Complex
}

#[repr(C)]
pub enum cufftResult_t {
    Success                 = 0,  //  The cuFFT operation was successful
    InvalidPlan             = 1,  //  cuFFT was passed an invalid plan handle
    AllocFailed             = 2,  //  cuFFT failed to allocate GPU or CPU memory
    InvalidType             = 3,  //  No longer used
    InvalidValue            = 4,  //  User specified an invalid pointer or parameter
    InternalError           = 5,  //  Driver or internal cuFFT library error
    ExecFailed              = 6,  //  Failed to execute an FFT on the GPU
    SetupFailed             = 7,  //  The cuFFT library failed to initialize
    InvalidSize             = 8,  //  User specified an invalid transform size
    UnalignedData           = 9,  //  No longer used
    IncompleteParameterList = 10, //  Missing parameters in call
    InvalidDevice           = 11, //  Execution of a plan was on different GPU than plan creation
    ParseError              = 12, //  Internal plan database error 
    NoWorkspace             = 13, //  No workspace has been provided prior to plan execution
    NotImplemented          = 14, // Function does not implement functionality for parameters given.
    LicenseError            = 15, // Used in previous versions.
    NotSupported            = 16  // Operation is not supported for parameters given.
}