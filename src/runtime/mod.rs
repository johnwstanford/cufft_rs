
use libc::size_t;

pub mod cuda_box;

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

