
use std::alloc::{alloc, Layout};

use libc::size_t;

use super::CudaMemCopyKind;

#[cfg(test)]
mod tests;

#[derive(Debug)]
pub struct CudaBox<T: Sized> {
	ptr: *mut T
}

impl<T: Sized> CudaBox<T> {

	pub fn new(x:T) -> Result<Self, &'static str> {
		let mut ptr:usize = 0;
		let src_ptr:*const T = &x;
		unsafe { 
			if super::cudaMalloc(&mut ptr, std::mem::size_of::<T>()) != 0 {
				return Err("Failure to allocate memory on device");
			}
			eprintln!("Allocated CudaBox(0x{:X}) with {} bytes", ptr as usize, std::mem::size_of::<T>());

			eprintln!("Copying from 0x{:X} to CudaBox(0x{:X})", src_ptr as usize, ptr);
			if super::cudaMemcpy(ptr as *mut u8, src_ptr as *const _, std::mem::size_of::<T>(), CudaMemCopyKind::HostToDevice) != 0 {
				return Err("Failure to copy memory to device");
			}	

		}

		eprintln!("Explicitly dropping value at 0x{:X}", &x as *const T as usize);
		std::mem::drop(x);

		Ok(Self{ ptr: ptr as *mut T })
	}

	pub fn to_host(self) -> Box<T> {
		let layout = Layout::new::<T>();
		unsafe {
			let dst:*mut u8 = alloc(layout);
			eprintln!("Copying from CudaBox(0x{:X}) to Box(0x{:X})", self.ptr as usize, dst as usize);
			if super::cudaMemcpy(dst, self.ptr as *const u8, std::mem::size_of::<T>(), CudaMemCopyKind::DeviceToHost) != 0 {
				panic!("Failure to copy memory from device");
			}
			Box::from_raw(dst as *mut T)
		}

	}

}

impl<T> std::ops::Drop for CudaBox<T> {

	fn drop(&mut self) {
		eprintln!("Dropping CudaBox(0x{:X})", self.ptr as usize);
		unsafe { super::cudaFree(self.ptr as size_t); }
	}

}

