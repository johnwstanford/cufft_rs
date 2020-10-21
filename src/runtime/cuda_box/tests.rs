
const N:usize = 4;

#[test]
fn allocate_deallocate_and_copy_device_memory() {

	let data0:[f32; N] = [1.4, 2.3, 3.2, 4.1];

	let dev_box = super::CudaBox::new(data0).unwrap();

	let host_box = dev_box.to_host();

	assert_eq!(*host_box, [1.4, 2.3, 3.2, 4.1]);

}