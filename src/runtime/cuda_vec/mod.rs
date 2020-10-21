
use libc::size_t;

use super::CudaMemCopyKind;

#[cfg(test)]
mod tests;

// TODO: think about whether or not we need Unique<T> here instead of *mut T.  Chapter 9 of the
// Rustnomicon uses it when describing how to implement Vec yourself, but I'm afraid that when Unique
// gets dropped, the host allocator might try to drop its memory, which might result in a seg fault 
// because the drop function would think it's on the heap, but it's on the CUDA device
#[derive(Debug)]
pub struct CudaVec<T: Sized + Clone> {
	ptr: *mut T,
	cap: usize,
	len: usize
}

impl<T: Sized + Clone> CudaVec<T> {

	//////////////
	// Functions that behave exactly the same as std::vec::Vec
	//////////////
	pub fn new() -> Self {
		Self{ ptr: 0 as *mut T, cap:0, len:0 }
	}

	pub fn as_mut_ptr(&self) -> *mut T   { self.ptr }
	pub fn as_ptr(&self)     -> *const T { self.ptr }
	pub fn len(&self)        -> usize    { self.len }
	pub fn capacity(&self)   -> usize    { self.cap }

	//////////////
	// Functions that behave a little differently from std::vec::Vec
	//////////////
	pub fn copy_from_slice(&mut self, src: &[T]) -> Result<(), &'static str> {
		// This is different from std::vec::Vec in that it will resize this vector
		// instead of panicking if it's the wrong size and it produces a Result return
		// type instead of ()
		self.set_capacity(src.len())?;
		let src_ptr:*const T = &src[0] as *const T;
		unsafe {
			super::cudaMemcpy(self.ptr as *mut u8, src_ptr as *const _, src.len()*std::mem::size_of::<T>(), CudaMemCopyKind::HostToDevice).ok()?;
		}
		self.len = src.len();

		Ok(())
	}	

	pub fn clone_to_host(&self) -> Result<Vec<T>, &'static str> {

		unsafe {
			let mut ans = vec![ std::mem::zeroed(); self.len];
			let dst:*mut T = &mut ans[0] as *mut T;
			super::cudaMemcpy(dst as *mut _, self.ptr as *const u8, self.len*std::mem::size_of::<T>(), CudaMemCopyKind::DeviceToHost).ok()?;
			Ok(ans)
		}

	}	

	pub fn set_capacity(&mut self, cap:usize) -> Result<(), &'static str> {

		// Whether or not there's new memory to allocate depends on      cap
		// Whether or not there's       data to     copy depends on self.len
		// Whether or not there's old memory to     free depends on self.cap

		if cap == self.cap { return Ok(()); } 		// Nothing to do

		if cap > 0 {
			// The requested capacity is nonzero, so we'll need to allocate new memory.  Even if we're reducing the capacity, we still
			// want to do this because it'll free up memory that we aren't using so someone else can use it

			// Allocate new memory
			let mut new_ptr:usize = 0;
			let new_byte_count:usize = std::mem::size_of::<T>() * cap;

			unsafe {
				super::cudaMalloc(&mut new_ptr, new_byte_count).ok()?;
				// eprintln!("Allocated {} bytes in CudaVec(0x{:X})", new_byte_count, new_ptr as usize);
			}

			if self.len > 0 {
				// Set the length first because you can truncate a vector by setting its capacity
				// to a smaller number than the length and if we're doing that, we don't want to 
				// copy those discarded bytes
				self.len = std::cmp::min(self.len, cap);

				// We've got old data to copy before we free the old memory
				unsafe { 

					// eprintln!("Copying from CudaVec's old allocation at 0x{:X} to its new allocation at 0x{:X}", self.ptr as usize, new_ptr);
					super::cudaMemcpy(new_ptr as *mut u8, self.ptr as *mut u8, std::mem::size_of::<T>() * self.len, CudaMemCopyKind::DeviceToDevice).ok()?;	

				}

			} 

			// Whether we copied old data into the new allocation or not, we're ready to free the old allocation, if applicable
			// Note that `self.cap` is the old capacity and `cap` is the new capacity 
			if self.cap > 0 {
				// Since the previous capacity was nonzero, we've got memory to free
				// eprintln!("Dropping old CudaVec allocation at 0x{:X}", self.ptr as usize);
				unsafe { super::cudaFree(self.ptr as size_t); }
			}

			self.ptr = new_ptr as *mut T;
			self.cap = cap;

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

	pub unsafe fn set_len(&mut self, new_len:usize) -> Result<(), &'static str> {
		if new_len < self.cap { 
			Err("Tried to set length longer than capacity") 
		} else {
			self.len = new_len;
			Ok(())
		}
	}

}

impl<T: Sized + Clone> std::ops::Drop for CudaVec<T> {

	fn drop(&mut self) {
		// eprintln!("Dropping CudaVec(0x{:X})", self.ptr as usize);
		unsafe { super::cudaFree(self.ptr as size_t); }
	}

}

