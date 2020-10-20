
use libc::size_t;

#[cfg(test)]
mod tests;

#[repr(C)]
pub enum CudaMemCopyKind {
	HostToHost     = 0,
	HostToDevice   = 1,
	DeviceToHost   = 2,
	DeviceToDevice = 3,
	Default        = 4		// Inferred from pointer types, requires unified virtual addressing
}

#[link(name = "cudart")]
extern {

	// Use size_t for pointers to device memory because it's simpler and also because it's an extra safeguard against
	// dereferencing them, even in unsafe blocks

	fn cudaMalloc(ptr:&mut size_t, size:size_t) -> i32;
	fn cudaMemcpy(dst:*mut u8, src:*const u8, count:size_t, kind:CudaMemCopyKind ) -> i32;
	fn cudaFree(ptr:size_t) -> i32;

}

unsafe fn memcpy_to_device<T: Sized>(dst:size_t, src:&[T]) -> Result<(), &'static str> {
	let src_ptr:*const T     = &src[0];
	let src_ptr_u8:*const u8 = src_ptr as *const _;
	match cudaMemcpy(dst as *mut u8, src_ptr_u8, std::mem::size_of::<T>()*src.len(), CudaMemCopyKind::HostToDevice) {
		0 => Ok(()),
		_ => Err("Unable to copy memory to device")
	}
}

unsafe fn memcpy_from_device<T: Sized>(dst:&mut [T], src:size_t) -> Result<(), &'static str> {
	let dst_ptr:*mut T     = &mut dst[0];
	let dst_ptr_u8:*mut u8 = dst_ptr as *mut _;
	match cudaMemcpy(dst_ptr_u8, src as *const u8, std::mem::size_of::<T>()*dst.len(), CudaMemCopyKind::DeviceToHost) {
		0 => Ok(()),
		_ => Err("Unable to copy memory from device")
	}
}

