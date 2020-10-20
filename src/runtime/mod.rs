
use libc::size_t;

#[link(name = "cudart")]
extern {

	pub fn cudaMalloc(ptr:&mut size_t, size:size_t) -> i32;
	pub fn cudaFree(ptr:size_t) -> i32;

}