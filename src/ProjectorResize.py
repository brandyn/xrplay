
from Projector import *

kernel_code = """
// 1:1 copy kernel - optimized for exact size match
extern "C" __global__
void rgb_to_rgba_copy(cudaSurfaceObject_t output,
                      const unsigned char* src, 
                      int width, int height, int src_pitch) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height)
        return;
    
    int src_idx = y * src_pitch + x * 3;
    
    uchar4 rgba = make_uchar4(
        src[src_idx + 0],  // R
        src[src_idx + 1],  // G
        src[src_idx + 2],  // B
        255                // A
    );
    
    surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
}

// Upscaling kernel - bilinear interpolation
extern "C" __global__
void rgb_to_rgba_upscale(cudaSurfaceObject_t output,
                          const unsigned char* src, 
                          int src_width, int src_height, int src_pitch,
                          int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height)
        return;
    
    // Map destination pixel to source coordinates
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    int ix0 = max(0, (int)floorf(src_x));
    int iy0 = max(0, (int)floorf(src_y));
    int ix1 = min(ix0 + 1, src_width - 1);
    int iy1 = min(iy0 + 1, src_height - 1);
    
    float fx = src_x - floorf(src_x);
    float fy = src_y - floorf(src_y);
    
    int idx00 = iy0 * src_pitch + ix0 * 3;
    int idx10 = iy0 * src_pitch + ix1 * 3;
    int idx01 = iy1 * src_pitch + ix0 * 3;
    int idx11 = iy1 * src_pitch + ix1 * 3;
    
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;
    
    float r = w00 * src[idx00] + w10 * src[idx10] + w01 * src[idx01] + w11 * src[idx11];
    float g = w00 * src[idx00+1] + w10 * src[idx10+1] + w01 * src[idx01+1] + w11 * src[idx11+1];
    float b = w00 * src[idx00+2] + w10 * src[idx10+2] + w01 * src[idx01+2] + w11 * src[idx11+2];
    
    uchar4 rgba = make_uchar4(
        (unsigned char)(r + 0.5f),
        (unsigned char)(g + 0.5f),
        (unsigned char)(b + 0.5f),
        255
    );
    
    surf2Dwrite(rgba, output, dst_x * sizeof(uchar4), dst_y);
}

// Downscaling kernel - area averaging (box filter)
extern "C" __global__
void rgb_to_rgba_downscale(cudaSurfaceObject_t output,
                            const unsigned char* src, 
                            int src_width, int src_height, int src_pitch,
                            int dst_width, int dst_height) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height)
        return;
    
    // Compute source region for this destination pixel
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    float src_x_start = dst_x * scale_x;
    float src_y_start = dst_y * scale_y;
    float src_x_end = src_x_start + scale_x;
    float src_y_end = src_y_start + scale_y;
    
    int x0 = (int)src_x_start;
    int y0 = (int)src_y_start;
    int x1 = min((int)ceilf(src_x_end), src_width);
    int y1 = min((int)ceilf(src_y_end), src_height);
    
    // Area averaging
    float r_sum = 0.0f, g_sum = 0.0f, b_sum = 0.0f;
    int sample_count = 0;
    
    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int idx = y * src_pitch + x * 3;
            r_sum += src[idx];
            g_sum += src[idx + 1];
            b_sum += src[idx + 2];
            sample_count++;
        }
    }
    
    if (sample_count > 0) {
        float inv_count = 1.0f / sample_count;
        uchar4 rgba = make_uchar4(
            (unsigned char)(r_sum * inv_count + 0.5f),
            (unsigned char)(g_sum * inv_count + 0.5f),
            (unsigned char)(b_sum * inv_count + 0.5f),
            255
        );
        surf2Dwrite(rgba, output, dst_x * sizeof(uchar4), dst_y);
    }
}
"""
class ResizeProjector(Projector):
    """A ResizeProjector just scales the image up or down (or does 1:1 copy) to fit the output.
    """
    def __init__(self):
        Projector.__init__(self)
        
        # Compile all three kernels
        mod                   = SourceModule(kernel_code)
        self.kernel_copy      = mod.get_function("rgb_to_rgba_copy")
        self.kernel_upscale   = mod.get_function("rgb_to_rgba_upscale")
        self.kernel_downscale = mod.get_function("rgb_to_rgba_downscale")
 
    def invoke(self, source_image, dest_image, dest_size, timers=None):
        
        # Ensure image is contiguous
        if not source_image.flags.c_contiguous:
            print("DEBUG: Source Image non-contiguous.")
            source_image = cp.ascontiguousarray(source_image)
        
        input_height, input_width, _ = source_image.shape
        output_width, output_height  = dest_size

        #timers and timers.push("map")

        # Map the graphics resource
        mapping = dest_image.map(self.cuda_stream)
        
        try:
            #timers and timers.start("mapping.array")

            # Get mapped array
            mapped_array = mapping.array(0, 0)
            
            if not mapped_array.handle:
                raise RuntimeError("Invalid mapped_array.handle (null)")
            
            #timers and timers.start("cudaResourceDesc")

            # Create resource descriptor
            res_desc = cudaResourceDesc()
            res_desc.resType = 0x00  # cudaResourceTypeArray
            res_desc.res.array.array = ctypes.c_void_p(int(mapped_array.handle))
            
            #timers and timers.start("createSurfaceObject")

            # Create surface object
            surf_obj = ctypes.c_ulonglong(0)
            err = self.cudaCreateSurfaceObject(ctypes.byref(surf_obj), ctypes.byref(res_desc))
            if err != 0:
                raise RuntimeError(f"Failed to create surface object: error {err}")
            
            #timers and timers.start("kernel")
            try:
                # Launch kernel - choose appropriate one based on scaling
                block_size = (16, 16, 1)
                grid_size = (
                    (output_width + block_size[0] - 1) // block_size[0],
                    (output_height + block_size[1] - 1) // block_size[1],
                    1
                )
                
                # Select kernel based on input vs output size
                if input_width == output_width and input_height == output_height:
                    # 1:1 copy - use optimized copy kernel
                    self.kernel_copy(
                        np.uint64(surf_obj.value),
                        np.intp(source_image.data.ptr),
                        np.int32(input_width),
                        np.int32(input_height),
                        np.int32(source_image.strides[0]),
                        block=block_size,
                        grid=grid_size,
                        stream=self.cuda_stream
                    )
                elif output_width > input_width or output_height > input_height:
                    # FIXME: untested case
                    # Upscaling - use bilinear interpolation
                    self.kernel_upscale(
                        np.uint64(surf_obj.value),
                        np.intp(source_image.data.ptr),
                        np.int32(input_width),
                        np.int32(input_height),
                        np.int32(source_image.strides[0]),
                        np.int32(output_width),
                        np.int32(output_height),
                        block=block_size,
                        grid=grid_size,
                        stream=self.cuda_stream
                    )
                else:
                    # Downscaling - use area averaging
                    self.kernel_downscale(
                        np.uint64(surf_obj.value),
                        np.intp(source_image.data.ptr),
                        np.int32(input_width),
                        np.int32(input_height),
                        np.int32(source_image.strides[0]),
                        np.int32(output_width),
                        np.int32(output_height),
                        block=block_size,
                        grid=grid_size,
                        stream=self.cuda_stream
                    )
                
                #timers and timers.start("sync")

                # Synchronize
                self.cuda_stream.synchronize()
            
            finally:
                #timers and timers.start("destroy_surf")

                # Destroy surface object
                self.cudaDestroySurfaceObject(surf_obj)
        
        finally:
            #timers and timers.start("unmap")
            # Unmap the texture
            mapping.unmap(self.cuda_stream)
    
            #timers and timers.pop()

