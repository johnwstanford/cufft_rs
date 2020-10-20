
pub mod runtime;

#[repr(C)]
pub enum cufftResult_t {
    Success                 = 0,  //  The cuFFT operation was successful
    InvalidPlan             = 1,  //  cuFFT was passed an invalid plan handle
    AllocFailed             = 2,  //  cuFFT failed to allocate GPU or CPU memory
    InvalidType             = 3,  //  No longer used
    InvalidValue            = 4,  //  User specified an invalid pointer or parameter
    InternalError           = 5,  //  Driver or internal cuFFT library error
    ExecFailed              = 6,  //  Failed to execute an FFT on the GPU
    SetupFailed             = 7,  //  The cuFFT library failed to initialize
    InvalidSize             = 8,  //  User specified an invalid transform size
    UnalignedData           = 9,  //  No longer used
    IncompleteParameterList = 10, //  Missing parameters in call
    InvalidDevice           = 11, //  Execution of a plan was on different GPU than plan creation
    ParseError              = 12, //  Internal plan database error 
    NoWorkspace             = 13, //  No workspace has been provided prior to plan execution
    NotImplemented          = 14, // Function does not implement functionality for parameters given.
    LicenseError            = 15, // Used in previous versions.
    NotSupported            = 16  // Operation is not supported for parameters given.
}