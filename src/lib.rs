
#[cfg(test)]
#[macro_use]
extern crate float_cmp;

pub mod runtime;
pub mod cufft;

#[repr(C)]
pub struct CError(i32);

impl CError {

	pub fn ok(&self) -> Result<(), &'static str> { match self.0 {
		0 => Ok(()),
		_ => Err("Failure in CError")
	}}

}

