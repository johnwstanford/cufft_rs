
use cufft_rs::runtime;

fn main() -> Result<(), &'static str> {

	let mut ptr:usize = 0;

	match unsafe { runtime::cudaMalloc(&mut ptr, std::mem::size_of::<f32>()) } {
		0 => (),
		_ => return Err("Unable to allocate memory")
	};

	match unsafe { runtime::cudaFree(ptr) } {
		0 => (),
		_ => return Err("Unable to free memory")
	}

	Ok(())

}