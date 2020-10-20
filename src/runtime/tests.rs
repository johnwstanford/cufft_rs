
use std::mem::size_of;

const N:usize = 4;

#[test]
fn allocate_deallocate_and_copy_device_memory() {

	let mut ptr:usize = 0;

	assert_eq!(0, unsafe { super::cudaMalloc(&mut ptr, N*size_of::<f32>()) });
	println!("Memory allocated on the device at {:X}", ptr);

	{
		let data0:[f32; N] = [1.4, 2.3, 3.2, 4.1];
		unsafe { super::memcpy_to_device(ptr, &data0).unwrap(); }
		println!("Sent {:?} to device", data0);
	}

	{
		let mut data1:[f32; N] = [0.0; N];
		unsafe { super::memcpy_from_device(&mut data1, ptr).unwrap(); }
		println!("Received {:?} from device", data1);
	}

	assert_eq!(0, unsafe { super::cudaFree(ptr) });
	
}