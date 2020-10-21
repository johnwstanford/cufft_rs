
use std::alloc::{alloc, Layout};

use libc::size_t;

use super::CudaMemCopyKind;

#[cfg(test)]
mod tests;

// TODO: think about whether or not we need Unique<T> here instead of *mut T.  Chapter 9 of the
// Rustnomicon uses it when describing how to implement Vec yourself, but I'm afraid that when Unique
// gets dropped, the host allocator might try to drop its memory, which might result in a seg fault 
// because the drop function would think it's on the heap, but it's on the CUDA device
#[derive(Debug)]
pub struct CudaVec<T: Sized> {
	ptr: *mut T,
	cap: usize,
	len: usize
}

impl<T: Sized> CudaVec<T> {

	pub fn new() -> Self {
		Self{ ptr: 0 as *mut T, cap:0, len:0 }
	}

	pub fn set_capacity(&mut self, cap:usize) -> Result<(), &'static str> {

		if cap == self.cap { return Ok(()); } 		// Nothing to do

		if cap > 0 {
			// The requested capacity is nonzero, so we'll need to allocate new memory.  Even if we're reducing the capacity, we still
			// want to do this because it'll free up memory that we aren't using so someone else can use it

			// Allocate new memory
			let mut new_ptr:usize = 0;
			let new_byte_count:usize = std::mem::size_of::<T>() * cap;
			unsafe {
				super::cudaMalloc(&mut new_ptr, new_byte_count).ok()?;
				eprintln!("Allocated {} bytes in CudaVec(0x{:X})", new_byte_count, new_ptr as usize);
			}

			if self.len > 0 {
				// We've got old data to copy before we free the old memory
				unsafe { 

					eprintln!("Copying from CudaVec's old allocation at 0x{:X} to its new allocation at 0x{:X}", self.ptr as usize, new_ptr);
					super::cudaMemcpy(new_ptr as *mut u8, self.ptr as *mut u8, std::mem::size_of::<T>() * self.len, CudaMemCopyKind::DeviceToDevice).ok()?;	

				}

			} 

			// Whether we copied old data into the new allocation or not, we're ready to free the old allocation, if applicable
			// Note that `self.cap` is the old capacity and `cap` is the new capacity 
			if self.cap > 0 {
				// Since the previous capacity was nonzero, we've got memory to free
				eprintln!("Dropping old CudaVec allocation at 0x{:X}", self.ptr as usize);
				unsafe { super::cudaFree(self.ptr as size_t); }
			}

			self.ptr = new_ptr as *mut T;
			self.cap = cap;
			// Length remains unchanged

		} else {
			// The requested capacity is zero, so there's no need to allocate new memory; just drop the old memory
			// if applicable and set the length and capacity to zero
			if self.cap > 0 {
				// Since the previous capacity was nonzero, we've got memory to free
				eprintln!("Dropping old CudaVec allocation at 0x{:X}", self.ptr as usize);
				unsafe { super::cudaFree(self.ptr as size_t); }
			}

			self.cap = 0;
			self.len = 0;
		}


		Ok(())
	}

	/*pub fn to_host(self) -> Box<T> {
		let layout = Layout::new::<T>();
		unsafe {
			let dst:*mut u8 = alloc(layout);
			eprintln!("Copying from CudaBox(0x{:X}) to Box(0x{:X})", self.ptr as usize, dst as usize);
			if super::cudaMemcpy(dst, self.ptr as *const u8, std::mem::size_of::<T>(), CudaMemCopyKind::DeviceToHost) != 0 {
				panic!("Failure to copy memory from device");
			}
			Box::from_raw(dst as *mut T)
		}

	}*/

}

impl<T> std::ops::Drop for CudaVec<T> {

	fn drop(&mut self) {
		eprintln!("Dropping CudaVec(0x{:X})", self.ptr as usize);
		unsafe { super::cudaFree(self.ptr as size_t); }
	}

}

