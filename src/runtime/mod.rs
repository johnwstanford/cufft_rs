
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

#[repr(C)]
#[derive(Debug, PartialEq)]
pub enum MemoryType {
    Unregistered = 0, /*< Unregistered memory */
    Host         = 1, /*< Host memory */
    Device       = 2, /*< Device memory */
    Managed      = 3, /*< Managed memory */
}

#[repr(C)]
#[derive(Debug)]
pub struct PointerAttributes {
    memory_typ:MemoryType,
    typ:MemoryType,
    device:i32,
    device_pointer:size_t,
    host_pointer:size_t,
    is_managed:i32
}

impl PointerAttributes {
	
	pub fn get(ptr:*const u8) -> Result<Self, &'static str> {
		unsafe { 
			let mut ans:Self = std::mem::zeroed();
			cudaPointerGetAttributes(&mut ans, ptr).ok()?;
			Ok(ans) 
		}
	}

}

#[link(name = "cudart")]
extern {

	// Use size_t for pointers to device memory because it's simpler and also because it's an extra safeguard against
	// dereferencing them, even in unsafe blocks

	fn cudaMalloc(ptr:&mut size_t, size:size_t) -> CError;
	fn cudaPointerGetAttributes(attributes:&mut PointerAttributes, ptr:*const u8) -> CError;
	fn cudaMemcpy(dst:*mut u8, src:*const u8, count:size_t, kind:CudaMemCopyKind ) -> CError;
	fn cudaFree(ptr:size_t) -> CError;

}

