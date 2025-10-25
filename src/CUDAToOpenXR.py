import ctypes

import numpy         as np
import pycuda.driver as cuda
import pycuda.gl     as cuda_gl

from pycuda.compiler import SourceModule
from pycuda.gl       import graphics_map_flags
from OpenGL          import GL

kernel_code = """
struct Mat3 {
    float m[9];  // Row-major 3x3 matrix
};

__device__ float3 normalize(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / len, v.y / len, v.z / len);
}

__device__ float3 mat3_mul_vec3(const Mat3& mat, float3 v) {
    return make_float3(
        mat.m[0] * v.x + mat.m[1] * v.y + mat.m[2] * v.z,
        mat.m[3] * v.x + mat.m[4] * v.y + mat.m[5] * v.z,
        mat.m[6] * v.x + mat.m[7] * v.y + mat.m[8] * v.z
    );
}

extern "C" __global__
void render_180_sbs(
    const unsigned char* input, int input_pitch, int input_width, int input_height,
    cudaSurfaceObject_t output, int output_width, int output_height,
    Mat3 rotation, int eye_index, 
    float tan_left, float tan_down, float tan_width, float tan_height,
    float inv_width, float inv_height, float u_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= output_width || y >= output_height)
        return;

    // Compute normalized screen coordinates using precomputed values
    float screen_x = tan_left + tan_width * ((float)x * inv_width);
    float screen_y = tan_down + tan_height * ((float)y * inv_height);

    // Create view ray (forward = -Z in OpenXR)
    float3 ray = normalize(make_float3(screen_x, screen_y, -1.0f));
    
    // Apply head rotation
    ray = mat3_mul_vec3(rotation, ray);

    // Convert to spherical coordinates (equirectangular)
    float theta = atan2f(ray.x, -ray.z);  // Azimuth: -pi to pi
    float phi   = asinf(ray.y);           // Elevation: -pi/2 to pi/2

    // For 180 SBS: map theta from [-pi/2, pi/2] to appropriate half
    float u = (theta * 0.318309886f + 0.5f) * 0.5f + u_offset;  // 1/pi = 0.318309886
    float v = 1.0f - (phi * 0.318309886f + 0.5f);  // 1/π = 0.318309886

    // Clamp UV coordinates
    u = fmaxf(0.0f, fminf(1.0f, u));
    v = fmaxf(0.0f, fminf(1.0f, v));
    
    // Convert to continuous pixel coordinates
    float src_x_f = u * input_width - 0.5f;
    float src_y_f = v * input_height - 0.5f;
    
    // Get integer coordinates and fractional parts for bilinear interpolation
    int x0 = max(0, (int)floorf(src_x_f));
    int y0 = max(0, (int)floorf(src_y_f));
    int x1 = min(x0 + 1, input_width - 1);
    int y1 = min(y0 + 1, input_height - 1);
    
    float fx = src_x_f - floorf(src_x_f);
    float fy = src_y_f - floorf(src_y_f);
    
    // Sample 4 corners
    int idx00 = y0 * input_pitch + x0 * 3;
    int idx10 = y0 * input_pitch + x1 * 3;
    int idx01 = y1 * input_pitch + x0 * 3;
    int idx11 = y1 * input_pitch + x1 * 3;
    
    // Bilinear interpolation for each channel
    float r = (1.0f - fx) * (1.0f - fy) * input[idx00] +
              fx * (1.0f - fy) * input[idx10] +
              (1.0f - fx) * fy * input[idx01] +
              fx * fy * input[idx11];
              
    float g = (1.0f - fx) * (1.0f - fy) * input[idx00 + 1] +
              fx * (1.0f - fy) * input[idx10 + 1] +
              (1.0f - fx) * fy * input[idx01 + 1] +
              fx * fy * input[idx11 + 1];
              
    float b = (1.0f - fx) * (1.0f - fy) * input[idx00 + 2] +
              fx * (1.0f - fy) * input[idx10 + 2] +
              (1.0f - fx) * fy * input[idx01 + 2] +
              fx * fy * input[idx11 + 2];

    // Convert to uchar4 and write
    uchar4 rgba = make_uchar4(
        (unsigned char)(r + 0.5f),
        (unsigned char)(g + 0.5f),
        (unsigned char)(b + 0.5f),
        255
    );
    surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
}
"""

# Proper CUDA resource descriptor with correct union size
class cudaResourceDesc(ctypes.Structure):
    class _ResUnion(ctypes.Union):
        class _ArrayType(ctypes.Structure):
            _fields_ = [('array', ctypes.c_void_p)]
        
        class _MipmappedArrayType(ctypes.Structure):
            _fields_ = [('mipmap', ctypes.c_void_p)]
        
        class _LinearType(ctypes.Structure):
            _fields_ = [
                ('devPtr', ctypes.c_void_p),
                ('desc', ctypes.c_byte * 32),  # cudaChannelFormatDesc
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

class CUDAToOpenXR:
    """Directly renders CUDA memory to swapchains of GL texture images (one chain per eye).
    This is the ultimate zero-copy path: decoder → swapchain.  (Note a "swapchain" here is
    nothing more than a list of texture images, usually just two or three, so that we can
    be rendering to one of them while the prior one is being sent to the next stage in the
    pipeline.  They need to be pre-allocated so the GL textures can be bound to cuda objects,
    but ultimately they are just lists of texture images.)
    
    Although called OpenXR, this class can be used to render to swapchains
        of any GL Texture images via various projection kernels.  For now the
        kernels are VR oriented (e.g. 180 SBS) but there's actually nothing
        OpenXR specific in this whole module.  It is, however, currently
        hard-coded to two views (left/right), but this could easily be relaxed.

    ALLOCATION:
      - Registers GL texture (typically OpenXR) swapchain images with CUDA
      - Does NOT allocate any intermediate buffers
    
    Init chain:
      - cuda and GL contexts must already be active
      - Requires GL swapchain images (typically from OpenXRDevice)
      - Call init(swapchain_images) where swapchain_images = 
        {'left': [gl_textures], 'right': [gl_textures]}
      - Returns None (terminal mapper)
    
    Runtime:
      - cuda and GL contexts must already be active
      - invoke(cuda_ptr, pitch, left_index, right_index, views)
    """
    
    def __init__(self, input_size, output_size, swapchain_images=None):

        self.width       , self.height        = input_size
        self.output_width, self.output_height = output_size
        
        # CUDA-registered swapchain images
        # Structure: {'left': [RegisteredImage...], 'right': [RegisteredImage...]}
        self.cuda_swapchains = {'left': [], 'right': []}
        self.cuda_stream     = None
        self.scaling_kernel  = None

        if swapchain_images is not None:
            self.init(swapchain_images)

    def init(self, swapchain_images):
        """
        Register all swapchain images with CUDA.
        
        Args:
            swapchain_images: {'left': [gl_textures], 'right': [gl_textures]}
        
        Returns:
            None (terminal mapper)
        """
        self.cuda_stream    = cuda.Stream()
        mod                 = SourceModule(kernel_code)
        self.render_kernel  = mod.get_function("render_180_sbs")    # Could look up function name based on requested transform.

        # Load CUDA runtime library for surface functions
        self.libcuda                           = ctypes.cdll.LoadLibrary("libcudart.so")
        self.cudaCreateSurfaceObject           = self.libcuda.cudaCreateSurfaceObject
        self.cudaCreateSurfaceObject.argtypes  = [
            ctypes.POINTER(ctypes.c_ulonglong),  # cudaSurfaceObject_t*
            ctypes.POINTER(cudaResourceDesc)     # cudaResourceDesc*
        ]
        self.cudaCreateSurfaceObject.restype   = ctypes.c_int
        self.cudaDestroySurfaceObject          = self.libcuda.cudaDestroySurfaceObject
        self.cudaDestroySurfaceObject.argtypes = [ctypes.c_ulonglong]
        self.cudaDestroySurfaceObject.restype  = ctypes.c_int

        # Register all swapchain images for both eyes
        for eye in ['left', 'right']:
            for gl_texture in swapchain_images[eye]:
                registered = cuda_gl.RegisteredImage(
                    int(gl_texture),
                    GL.GL_TEXTURE_2D,
                    graphics_map_flags.WRITE_DISCARD
                )
                self.cuda_swapchains[eye].append(registered)
    
    def invoke(self, cuda_ptr, pitch, left_index, right_index, views=None, leveling_offset=0):
        """
        Render from cuda memory straight to output GL texture (swapchain image)
            with the currently selected kernel (e.g., 180 SBS), using view
            poses (usually OpenXR Views) corresponding to each output where applicable.
        
        Args:
            cuda_ptr: Device pointer to RGB input data
            pitch: Input pitch in bytes
            left_index: Left eye swapchain image index
            right_index: Right eye swapchain image index
            views: render orientation for each view --
                    list of 2 view (usually xr.View) objects with .pose and .fov
            leveling_offset: radians of rotational pitch adjustment.  Positive tilts scene down.
        """
        if views is None or len(views) < 2:
            raise ValueError("180 SBS rendering requires 2 views")
        
        for eye_idx, eye_name, swapchain_index, view in [
            (0,  'left',  left_index, views[0]), 
            (1, 'right', right_index, views[1])
        ]:
            registered_image = self.cuda_swapchains[eye_name][swapchain_index]

            # Extract rotation matrix from quaternion and convert to rotation matrix (3x3)
            quat       = view.pose.orientation
            rot_matrix = self._quat_to_matrix(quat.x, quat.y, quat.z, quat.w)

            # Apply pitch leveling offset, if applicable
            if leveling_offset:

                cos_p = np.cos(leveling_offset)
                sin_p = np.sin(leveling_offset)
                
                # Rotation matrix for pitch (rotation around X-axis)
                pitch_matrix = np.array([
                    [1,     0,      0],
                    [0, cos_p, -sin_p],
                    [0, sin_p,  cos_p]
                ])
                
                # Combine rotations: first apply the head rotation, then the pitch offset
                rot_matrix = pitch_matrix @ rot_matrix

            # Map the swapchain image
            mapping = registered_image.map(self.cuda_stream)

            try:
                # Get mapped array (mipmap level 0, layer 0)
                mapped_array = mapping.array(0, 0)

                if not mapped_array.handle:
                    raise RuntimeError(f"Invalid mapped_array.handle for eye {eye_name} (null)")
                
                # Create resource descriptor
                res_desc                 = cudaResourceDesc()
                res_desc.resType         = 0x00  # cudaResourceTypeArray
                res_desc.res.array.array = ctypes.c_void_p(int(mapped_array.handle))

                # Create surface object
                surf_obj = ctypes.c_ulonglong(0)
                err      = self.cudaCreateSurfaceObject(ctypes.byref(surf_obj), ctypes.byref(res_desc))
                if err  != 0:
                    raise RuntimeError(f"Failed to create surface object for eye {eye_name}: error {err}")
                
                try:
                    # Launch CUDA kernel
                    block_size = (16, 16, 1)
                    grid_size = (
                        (self.output_width  + block_size[0] - 1) // block_size[0],
                        (self.output_height + block_size[1] - 1) // block_size[1],
                        1
                    )

                    # Prepare rotation matrix as flat array
                    rot_array = np.array(rot_matrix, dtype=np.float32).flatten()
                    
                    # Precompute FOV tangent values
                    fov        = view.fov
                    tan_left   = np.tan(fov.angle_left)
                    tan_right  = np.tan(fov.angle_right)
                    tan_up     = np.tan(fov.angle_up)
                    tan_down   = np.tan(fov.angle_down)
                    tan_width  = tan_right - tan_left
                    tan_height = tan_up - tan_down
                    inv_width  = 1.0 / self.output_width
                    inv_height = 1.0 / self.output_height
                    u_offset   = 0.0 if eye_idx == 0 else 0.5
                    
                    self.render_kernel(
                        np.intp(cuda_ptr),
                        np.int32(pitch),
                        np.int32(self.width),
                        np.int32(self.height),
                        np.uint64(surf_obj.value),
                        np.int32(self.output_width),
                        np.int32(self.output_height),
                        rot_array,                      # 3x3 rotation matrix (9 floats)
                        np.int32(eye_idx),              # Eye index (0=left, 1=right)
                        np.float32(tan_left),
                        np.float32(tan_down),
                        np.float32(tan_width),
                        np.float32(tan_height),
                        np.float32(inv_width),
                        np.float32(inv_height),
                        np.float32(u_offset),
                        block=block_size,
                        grid=grid_size,
                        stream=self.cuda_stream
                    )

                    # Synchronize the stream
                    self.cuda_stream.synchronize()

                finally:
                    # Destroy surface object
                    self.cudaDestroySurfaceObject(surf_obj)
                    
            finally:
                # Unmap the texture
                mapping.unmap(self.cuda_stream)
    
    def _quat_to_matrix(self, x, y, z, w):
        """Convert quaternion to 3x3 rotation matrix (row-major)."""
        return np.array([
           [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
           [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
           [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    def close(self):
        """Unregister all CUDA resources."""
        for eye in ['left', 'right']:
            for registered in self.cuda_swapchains[eye]:
                registered.unregister()
        
        if self.cuda_stream:
            self.cuda_stream.synchronize()

