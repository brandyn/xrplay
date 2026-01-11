
import ctypes

# CUDA resource descriptor
class cudaResourceDesc(ctypes.Structure):
    class _ResUnion(ctypes.Union):
        class _ArrayType(ctypes.Structure):
            _fields_ = [('array', ctypes.c_void_p)]
        
        class _MipmappedArrayType(ctypes.Structure):
            _fields_ = [('mipmap', ctypes.c_void_p)]
        
        class _LinearType(ctypes.Structure):
            _fields_ = [
                ('devPtr', ctypes.c_void_p),
                ('desc', ctypes.c_byte * 32),
                ('sizeInBytes', ctypes.c_size_t)
            ]
        
        class _PitchType(ctypes.Structure):
            _fields_ = [
                ('devPtr', ctypes.c_void_p),
                ('desc', ctypes.c_byte * 32),
                ('width', ctypes.c_size_t),
                ('height', ctypes.c_size_t),
                ('pitchInBytes', ctypes.c_size_t)
            ]
        
        _fields_ = [
            ('array', _ArrayType),
            ('mipmap', _MipmappedArrayType),
            ('linear', _LinearType),
            ('pitch', _PitchType)
        ]
    
    _fields_ = [
        ('resType', ctypes.c_int),
        ('res', _ResUnion)
    ]

# Load CUDA runtime library for surface functions
libcuda = ctypes.cdll.LoadLibrary("libcudart.so")

