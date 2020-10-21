
use libc::size_t;

use super::CError;

pub mod cuda_box;
pub mod cuda_vec;

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

	fn cudaMalloc(ptr:&mut size_t, size:size_t) -> CError;
	fn cudaMemcpy(dst:*mut u8, src:*const u8, count:size_t, kind:CudaMemCopyKind ) -> CError;
	fn cudaFree(ptr:size_t) -> CError;

}

