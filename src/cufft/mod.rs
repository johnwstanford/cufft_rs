
use libc::size_t;

use crate::CError;
use crate::runtime::cuda_vec::CudaVec;

#[cfg(test)]
mod tests;

#[link(name = "cufft")]
extern {

    fn cufftPlanMany(plan:&mut size_t, rank:i32, n:&i32, inembed:&i32, istride:i32, idist:i32,
        onembed:&i32, ostride:i32, odist:i32, fft_type:cufftType_t, batch:i32) -> cufftResult_t;

    // TODO: Make these pointers a *const Complex<f64> and *mut Complex<f64> then use the type parameter on PlanComplex1D
    // to pick the right function
	fn cufftExecZ2Z(plan:size_t, idata:*const u8, odata:*mut u8, direction:Direction) -> CError;

    fn cufftDestroy(plan:size_t) -> cufftResult_t;

}

#[derive(Debug)]
pub struct PlanComplex1D<T: Clone + Default> {
    handle:usize,
    n:i32,          // --- Size of the Fourier transform
    batch_count:i32,
    buffer:CudaVec<Complex<T>>
}

impl<T: Clone + Default> PlanComplex1D<T> {

    pub fn new(n:i32, batch_count:i32) -> Result<Self, &'static str> {
        if batch_count != 1 { return Err("TODO: implement batch counts other than 1"); }

        let mut handle:usize = 0;

        let rank:i32 = 1; // --- 1D FFTs

        // --- Distance between two successive input/output elements
        let istride:i32 = 1;
        let ostride:i32 = 1;
        
        let inembed:[i32; 1] = [0];                  // --- Input size with pitch (ignored for 1D transforms)
        let onembed:[i32; 1] = [0];                  // --- Output size with pitch (ignored for 1D transforms)

    	let buffer = CudaVec::from_slice(&vec![Complex(T::default(), T::default()); n as usize])?;

        match unsafe { cufftPlanMany(&mut handle, rank, &n, 
                &inembed[0], istride, n,
                &onembed[0], ostride, n, 
                cufftType_t::Z2Z, batch_count) } {
            cufftResult_t::Success => Ok(PlanComplex1D{ handle, n, batch_count, buffer }),
            _                      => Err("Unable to create FFT plan")
        }
        
    }

    pub fn fwd(&mut self, time_domain:&CudaVec<Complex<T>>) -> Result<Vec<Complex<T>>, &'static str> {
    	if time_domain.len() != self.n as usize { 
    		Err("Wrong sized input") 
    	} else {
	    	unsafe { cufftExecZ2Z(self.handle, time_domain.as_ptr() as *const _, self.buffer.as_mut_ptr() as *mut _, Direction::Forward).ok()?; }
	    	Ok(self.buffer.clone_to_host()?)		
    	}
    }

}

impl<T: Clone + Default> std::ops::Drop for PlanComplex1D<T> {

    fn drop(&mut self) {
        unsafe { cufftDestroy(self.handle); }
    }

}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Complex<T>(T, T);

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