from Projector import *
from math      import tau

vr_kernel_code = """
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

#ifdef ENABLE_CHROMAKEY

struct ChromakeyParams {
    float key_cb_ratio, key_cr_ratio;   // Key color in Cb/Y, Cr/Y ratio space
    float threshold_inner;              // Distance below which alpha=0 (fully keyed)
    float threshold_outer;              // Distance above which alpha=1 (fully opaque)
    float y_gate_min;                   // Luminance below which keying is disabled     (Pass darks)
    float y_gate_max;                   // Luminance above which keying is fully active (Pass darks)
    float y_gate2_min;                  // Luminance above which keying is disabled     (Pass lights)
    float y_gate2_max;                  // Luminance below which keying is fully active (Pass lights)
};

__device__ float smoothstep(float edge0, float edge1, float x) {
    float t = fmaxf(0.0f, fminf(1.0f, (x - edge0) / (edge1 - edge0)));
    return t * t * (3.0f - 2.0f * t);
}

__device__ __inline__ float compute_chromakey_alpha(float r, float g, float b, const ChromakeyParams params) {
    
    // Convert to 0-1 range
    float rf = r / 255.0f;
    float gf = g / 255.0f;
    float bf = b / 255.0f;
    
    // RGB to YCbCr (BT.709)
    float Y  =  0.2126f * rf + 0.7152f * gf + 0.0722f * bf;
    float Cb = -0.1146f * rf - 0.3854f * gf + 0.5000f * bf;
    float Cr =  0.5000f * rf - 0.4542f * gf - 0.0458f * bf;
    
    // Luminance gate: fade out keying at low and high brightness
    float y_gate =    smoothstep(params.y_gate_min , params.y_gate_max , Y);
    y_gate *= (1.0f - smoothstep(params.y_gate2_min, params.y_gate2_max, Y));

    // Compute chrominance ratios (normalized by luminance)
    float inv_Y    = 1.0f / fmaxf(Y, 0.001f);  // Avoid division by zero
    float cb_ratio = Cb * inv_Y;
    float cr_ratio = Cr * inv_Y;
    
    // Distance in ratio space from key color ratios
    float dcb = cb_ratio - params.key_cb_ratio;
    float dcr = cr_ratio - params.key_cr_ratio;
    float dist = sqrtf(dcb*dcb + dcr*dcr);
    
    // Compute chroma-based alpha
    float chroma_alpha = smoothstep(params.threshold_inner, params.threshold_outer, dist);
    
    // Combine with luminance gate
    // Low luminance (y_gate=0): alpha=1 (opaque, preserve dark objects)
    // High luminance (y_gate=1): alpha=chroma_alpha (full keying)
    return chroma_alpha * y_gate + (1.0f - y_gate);
}

#endif

extern "C" __global__
void render_vr_projection(
    const unsigned char* input, int input_pitch, int input_width, int input_height,
    cudaSurfaceObject_t output, int output_width, int output_height,
    Mat3 rotation,
    float tan_left, float tan_down, float tan_width, float tan_height,
    float inv_width, float inv_height,
    float theta_scale, float u_scale, float v_scale, float u_offset, float v_offset
#ifdef ENABLE_CHROMAKEY
    , const ChromakeyParams chromakey
#endif
    )
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

    // Parameterized UV mapping for different projection types
    float u = (theta * theta_scale + 0.5f) * u_scale + u_offset;
    float v = (1.0f - (phi * 0.318309886f + 0.5f)) * v_scale + v_offset;  // 1/π = 0.318309886

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

#ifdef ENABLE_CHROMAKEY
    // Compute alpha from chromakey
    float alpha = compute_chromakey_alpha(r, g, b, chromakey);
#else
    float alpha = 1.0;
#endif

    // Convert to uchar4 and write
    uchar4 rgba = make_uchar4(
#ifdef DEBUG_CHROMAKEY
        (unsigned char)(alpha * 255.99f),
        (unsigned char)(128.0),
        (unsigned char)((1-alpha) * 255.99f),
#else
        (unsigned char)(r + 0.5f),
        (unsigned char)(g + 0.5f),
        (unsigned char)(b + 0.5f),
#endif
        (unsigned char)(alpha * 255.99f)
    );
    surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
}

extern "C" __global__
void render_fisheye_projection(
    const unsigned char* input, int input_pitch, int input_width, int input_height,
    cudaSurfaceObject_t output, int output_width, int output_height,
    Mat3 rotation,
    float tan_left, float tan_down, float tan_width, float tan_height,
    float inv_width, float inv_height,
    float max_angle, float u_offset
#ifdef ENABLE_CHROMAKEY
    , const ChromakeyParams chromakey
#endif
    )
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

    // Convert to fisheye coordinates
    // Compute angle from forward direction (-Z axis)
    float theta = acosf(-ray.z);  // Angle from forward, 0 to pi
    
    // Check if outside fisheye field of view
    if (theta > max_angle) {
        // Outside fisheye FOV, render transparent black
        uchar4 rgba = make_uchar4(0, 0, 0, 0);
        surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
        return;
    }
    
    // Compute azimuthal angle
    float phi = atan2f(ray.x, ray.y);  // Angle around forward axis
    
    // Map theta to radial distance (0 at center, 1 at edge)
    float radius = theta / max_angle;
    
    // Convert polar to Cartesian UV (centered at 0.5, 0.5)
    float u_local = 0.5f + radius * sinf(phi) * 0.5f;
    float v_local = 0.5f - radius * cosf(phi) * 0.5f;
    
    // Apply side-by-side offset (scale to half width and offset)
    float u = u_local * 0.5f + u_offset;
    float v = v_local;
    
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
    float red = (1.0f - fx) * (1.0f - fy) * input[idx00] +
                fx * (1.0f - fy) * input[idx10] +
                (1.0f - fx) * fy * input[idx01] +
                fx * fy * input[idx11];
              
    float green = (1.0f - fx) * (1.0f - fy) * input[idx00 + 1] +
                  fx * (1.0f - fy) * input[idx10 + 1] +
                  (1.0f - fx) * fy * input[idx01 + 1] +
                  fx * fy * input[idx11 + 1];
              
    float blue = (1.0f - fx) * (1.0f - fy) * input[idx00 + 2] +
                 fx * (1.0f - fy) * input[idx10 + 2] +
                 (1.0f - fx) * fy * input[idx01 + 2] +
                 fx * fy * input[idx11 + 2];

#ifdef ENABLE_CHROMAKEY
    // Compute alpha from chromakey
    float alpha = compute_chromakey_alpha(red, green, blue, chromakey);
#else
    float alpha = 1.0;
#endif

    // Convert to uchar4 and write
    uchar4 rgba = make_uchar4(
#ifdef DEBUG_CHROMAKEY
        (unsigned char)(alpha * 255.99f),
        (unsigned char)(128.0),
        (unsigned char)((1-alpha) * 255.99f),
#else
        (unsigned char)(red + 0.5f),
        (unsigned char)(green + 0.5f),
        (unsigned char)(blue + 0.5f),
#endif
        (unsigned char)(alpha * 255.99f)
    );
    surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
}

extern "C" __global__
void render_flat_projection(
    const unsigned char* input, int input_pitch, int input_width, int input_height,
    cudaSurfaceObject_t output, int output_width, int output_height,
    Mat3 rotation,
    float tan_left, float tan_down, float tan_width, float tan_height,
    float inv_width, float inv_height,
    float screen_width, float screen_height, float screen_distance,
    float u_scale, float u_offset
    )
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

    // Intersect ray with virtual screen plane at distance screen_distance
    // Screen plane is perpendicular to -Z axis, centered at (0, 0, -screen_distance)
    float t = -screen_distance / ray.z;
    
    // Check if ray hits the plane (t must be positive)
    if (t <= 0.0f) {
        // Ray doesn't hit the screen, render transparent black
        uchar4 rgba = make_uchar4(0, 0, 0, 0);
        surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
        return;
    }
    
    // Compute intersection point
    float hit_x = ray.x * t;
    float hit_y = ray.y * t;
    
    // Check if hit point is within screen bounds
    float half_width = screen_width * 0.5f;
    float half_height = screen_height * 0.5f;
    
    if (hit_x < -half_width || hit_x > half_width ||
        hit_y < -half_height || hit_y > half_height) {
        // Outside screen bounds, render transparent black
        uchar4 rgba = make_uchar4(0, 0, 0, 0);
        surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
        return;
    }
    
    // Convert hit point to UV coordinates (0 to 1)
    float u = (hit_x / screen_width + 0.5f) * u_scale + u_offset;  // 0.5 u_scale for side-by-side
    float v = 1.0f - (hit_y / screen_height + 0.5f);
    
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

    // Convert to uchar4 and write (fully opaque)
    uchar4 rgba = make_uchar4(
        (unsigned char)(r + 0.5f),
        (unsigned char)(g + 0.5f),
        (unsigned char)(b + 0.5f),
        255
    );
    surf2Dwrite(rgba, output, x * sizeof(uchar4), y);
}
"""

class VRProjector(Projector):

    """Handles VR projection (180/etc/360, SBS/TB, fisheye, flat) to linear projection based on a viewing matrix.
    """
    def __init__(self, projection='180', chromakey=None):
        """
        Args:
            projection: '360' for 360° top-bottom, any other degrees for side-by-side, 
                       'F{degrees}' for fisheye (e.g., 'F180'), 'flat' for virtual 3d cinema,
                       'mono' for virtual 2d cinema.
            chromakey: Optional dict with keys:
                - 'key_color': (r, g, b) in 0-255 range (default blue: (0, 0, 255))
                - 'threshold_inner': float (default 0.1)
                - 'threshold_outer': float (default 0.3)
                - 'y_gate_min': float (default 0.05)
                - 'y_gate_max': float (default 0.15)
                - 'y_gate2_min': float (default 0.05)
                - 'y_gate2_max': float (default 0.15)
                If None, chromakey is disabled.
        """
        Projector.__init__(self)

        # Set up chromakey parameters
        self.set_chromakey(chromakey)
        
        # Set up projection parameters
        self.set_projection(projection)
        
        # Compile kernels - with and without chromakey
        mod = SourceModule(vr_kernel_code)
        self.render_kernel      = mod.get_function("render_vr_projection")
        self.render_fisheye_kernel = mod.get_function("render_fisheye_projection")
        self.render_flat_kernel = mod.get_function("render_flat_projection")
        
        # Compile chromakey version separately
        mod_chromakey = SourceModule(vr_kernel_code, options=['-DENABLE_CHROMAKEY'])
        self.render_kernel_chromakey = mod_chromakey.get_function("render_vr_projection")
        self.render_fisheye_kernel_chromakey = mod_chromakey.get_function("render_fisheye_projection")

        self.fov = None # Cache for naive invoke() mode

    def set_projection(self, projection):
        """Set projection type. Can be changed on the fly.  See __init__ docs for valid projections.
        """
        if projection == '360':
            # 360° top-bottom
            self.theta_scale = 0.159154943  # 1/tau     -- radians -> u coordinate
            self.u_scale = 1.0
            self.v_scale = 0.5
            self.u_offset_base = 0.0
            self.v_offset_base = 0.5  # For eye selection
            self.is_fisheye = False
        elif projection.startswith('F') and projection[1:].replace('.','',1).isdigit():
            # Fisheye projection (e.g., 'F180')
            degrees = float(projection[1:])
            self.max_angle = np.radians(degrees / 2)  # Half angle for fisheye
            self.u_offset_base = 0.5  # For eye selection (side-by-side)
            self.is_fisheye = True
        elif projection.isdecimal():    # Assume side-by-side for anything by 360 for now.
            # Any degrees side-by-side:
            self.theta_scale = 360/(tau * float(projection))
            self.u_scale = 0.5
            self.v_scale = 1.0
            self.u_offset_base = 0.5  # For eye selection
            self.v_offset_base = 0.0
            self.is_fisheye = False
        elif projection == 'flat':
            # Flat virtual 3d cinema screen (side-by-side)
            self.u_scale       = 0.5    # Half the image for each eye
            self.u_offset_base = 0.5    # For eye selection (0=left, 0.5=right)
            self.is_fisheye    = False
        elif projection == 'mono':
            # Flat virtual 2d cinema screen
            self.u_scale       = 1.0    # Whole image for each eye
            self.u_offset_base = 0.0    # Same source both eyes.
            self.is_fisheye    = False
        else:
            raise ValueError(f"Unknown projection type: {projection}. Use '360' (top/bot), '180' or other degrees (SBS), 'F180' or other degrees (fisheye), 'flat' or 'mono'.")
        
        self.projection = projection

    def set_chromakey(self, chromakey):
        """Set chromakey configuration. Pass None to disable chromakey.
        
        Args:
            chromakey: Dict with optional keys:
                - 'key_color': (r, g, b) in 0-255 range (default blue: (0, 0, 255))
                - 'threshold_inner': float (default 0.10)
                - 'threshold_outer': float (default 0.11)
                - 'y_gate_min': float (default 0.03)
                - 'y_gate_max': float (default 0.05)
                - 'y_gate2_min': float (default 0.20)
                - 'y_gate2_max': float (default 0.22)
                Or None to disable chromakey.
        """
        if chromakey is None:
            self.chromakey_array = None
        else:
            key_color = chromakey.get('key_color', (0, 0, 255))  # Default: blue
            
            # Convert key color to YCbCr ratios (BT.709) - no division by 255 needed for ratios
            kr, kg, kb = float(key_color[0]), float(key_color[1]), float(key_color[2])
            key_Y  =  0.2126 * kr + 0.7152 * kg + 0.0722 * kb
            key_Cb = -0.1146 * kr - 0.3854 * kg + 0.5000 * kb
            key_Cr =  0.5000 * kr - 0.4542 * kg - 0.0458 * kb
            key_cb_ratio = key_Cb / max(key_Y, 0.1)
            key_cr_ratio = key_Cr / max(key_Y, 0.1)
            
            self.chromakey_array = np.array([
                key_cb_ratio,
                key_cr_ratio,
                chromakey.get('threshold_inner', 0.10),
                chromakey.get('threshold_outer', 0.11),
                chromakey.get('y_gate_min', 0.03),
                chromakey.get('y_gate_max', 0.05),
                chromakey.get('y_gate2_min', 0.20),
                chromakey.get('y_gate2_max', 0.22),
            ], dtype=np.float32)

    def apply_view_adjustments(self, rot_matrix, leveling_offset=0, yaw_offset=0):
        if yaw_offset:
            cos_p = np.cos(yaw_offset)
            sin_p = np.sin(yaw_offset)
            # Rotation matrix for yaw (rotation around Y-axis)
            yaw_matrix = np.array([
                [cos_p, 0, -sin_p],
                [0,     1,      0],
                [sin_p, 0,  cos_p]
            ])
            rot_matrix = yaw_matrix @ rot_matrix

        if leveling_offset:
            cos_p = np.cos(leveling_offset)
            sin_p = np.sin(leveling_offset)
            # Rotation matrix for pitch (rotation around X-axis)
            pitch_matrix = np.array([
                [1,     0,      0],
                [0, cos_p, -sin_p],
                [0, sin_p,  cos_p]
            ])
            rot_matrix = pitch_matrix @ rot_matrix

        return rot_matrix

    def invoke_vr(self, src_image, dst_image, dst_size, rot_matrix, fov, leveling_offset=0, yaw_offset=0, eye=0, screen_width=4.4, screen_height=2.475, screen_distance=3.0):
            """Project a single, angularly mapped (VR style) src_image to linear dst_image.

            src_image   is a cupy RGB image.
            dst_image   must be registered with cuda such that dst_image.map(self.cuda_stream) will work.
            rot_matrix  is a 3x3 matrix which orients the scene to the camera.  (See usage below for exact meaning.)
            fov         is an OpenXR style view.fov, with angles given in radians for:
              .angle_left
              .angle_right
              .angle_up
              .angle_down
            leveling_offset and yaw_offset are radians to pre-adjust rot_matrix's pitch and yaw by, as a convenience.
            eye         is 0 or 1, selecting which eye's view to render (0=first, 1=second).
                        For SBS (incl flat, fisheye): 0=left half, 1=right half
                        For (360°) TB: 0=top half, 1=bottom half
            screen_width  is the width of the virtual screen in meters      (flat/mono projections only, default 4.4m)
            screen_height is the height of the virtual screen in meters     (flat/mono projections only, default 2.475m)
            screen_distance is the distance from viewer to screen in meters (flat/mono projections only, default 3.0m)
            """
            # Apply pitch leveling and yaw offsets, if applicable
            rot_matrix = self.apply_view_adjustments(rot_matrix, leveling_offset, yaw_offset)

            mapping = dst_image.map(self.cuda_stream)

            try:
                # Get mapped array (mipmap level 0, layer 0)
                mapped_array = mapping.array(0, 0)

                if not mapped_array.handle:
                    raise RuntimeError(f"Invalid mapped_array.handle (null)")
                
                # Create resource descriptor
                res_desc                 = cudaResourceDesc()
                res_desc.resType         = 0x00  # cudaResourceTypeArray
                res_desc.res.array.array = ctypes.c_void_p(int(mapped_array.handle))

                # Create surface object
                surf_obj = ctypes.c_ulonglong(0)
                err      = self.cudaCreateSurfaceObject(ctypes.byref(surf_obj), ctypes.byref(res_desc))
                if err  != 0:
                    raise RuntimeError(f"Failed to create surface object: error {err}")
                
                try:
                    input_height, input_width, _ = src_image.shape
                    output_width, output_height  = dst_size

                    # Launch CUDA kernel (need to reduce block size on chromakey because of register pressure)
                    block_size = (16, 16, 1)
                    grid_size = (
                        (output_width  + block_size[0] - 1) // block_size[0],
                        (output_height + block_size[1] - 1) // block_size[1],
                        1
                    )

                    # Prepare rotation matrix as flat array
                    rot_array = np.array(rot_matrix, dtype=np.float32).flatten()
                    
                    # Precompute FOV tangent values
                    tan_left   = np.tan(fov.angle_left)
                    tan_right  = np.tan(fov.angle_right)
                    tan_up     = np.tan(fov.angle_up)
                    tan_down   = np.tan(fov.angle_down)
                    tan_width  = tan_right - tan_left
                    tan_height = tan_up - tan_down
                    inv_width  = 1.0 / output_width
                    inv_height = 1.0 / output_height
                    
                    # Choose kernel based on projection type
                    if self.projection in ('flat', 'mono'):
                        # Flat projection: compute screen dimensions from size parameter
                        u_offset = self.u_offset_base * eye
                        
                        self.render_flat_kernel(
                            np.intp(src_image.data.ptr),
                            np.int32(src_image.strides[0]),
                            np.int32(input_width),
                            np.int32(input_height),
                            np.uint64(surf_obj.value),
                            np.int32(output_width),
                            np.int32(output_height),
                            rot_array,
                            np.float32(tan_left),
                            np.float32(tan_down),
                            np.float32(tan_width),
                            np.float32(tan_height),
                            np.float32(inv_width),
                            np.float32(inv_height),
                            np.float32(screen_width),
                            np.float32(screen_height),
                            np.float32(screen_distance),
                            np.float32(self.u_scale),
                            np.float32(u_offset),
                            block=block_size,
                            grid=grid_size,
                            stream=self.cuda_stream
                        )
                    elif self.is_fisheye:
                        # Fisheye projection
                        u_offset = self.u_offset_base * eye
                        
                        # Choose kernel based on chromakey state
                        if self.chromakey_array is not None:
                            self.render_fisheye_kernel_chromakey(
                                np.intp(src_image.data.ptr),
                                np.int32(src_image.strides[0]),
                                np.int32(input_width),
                                np.int32(input_height),
                                np.uint64(surf_obj.value),
                                np.int32(output_width),
                                np.int32(output_height),
                                rot_array,
                                np.float32(tan_left),
                                np.float32(tan_down),
                                np.float32(tan_width),
                                np.float32(tan_height),
                                np.float32(inv_width),
                                np.float32(inv_height),
                                np.float32(self.max_angle),
                                np.float32(u_offset),
                                self.chromakey_array,
                                block=block_size,
                                grid=grid_size,
                                stream=self.cuda_stream
                            )
                        else:
                            self.render_fisheye_kernel(
                                np.intp(src_image.data.ptr),
                                np.int32(src_image.strides[0]),
                                np.int32(input_width),
                                np.int32(input_height),
                                np.uint64(surf_obj.value),
                                np.int32(output_width),
                                np.int32(output_height),
                                rot_array,
                                np.float32(tan_left),
                                np.float32(tan_down),
                                np.float32(tan_width),
                                np.float32(tan_height),
                                np.float32(inv_width),
                                np.float32(inv_height),
                                np.float32(self.max_angle),
                                np.float32(u_offset),
                                block=block_size,
                                grid=grid_size,
                                stream=self.cuda_stream
                            )
                    else:
                        # Equirectangular projection (360° or degrees SBS)
                        # Compute UV offsets based on eye selection
                        u_offset = self.u_offset_base * eye
                        v_offset = self.v_offset_base * eye

                        # Choose kernel based on chromakey state
                        if self.chromakey_array is not None:
                            self.render_kernel_chromakey(
                                np.intp(src_image.data.ptr),
                                np.int32(src_image.strides[0]),
                                np.int32(input_width),
                                np.int32(input_height),
                                np.uint64(surf_obj.value),
                                np.int32(output_width),
                                np.int32(output_height),
                                rot_array,
                                np.float32(tan_left),
                                np.float32(tan_down),
                                np.float32(tan_width),
                                np.float32(tan_height),
                                np.float32(inv_width),
                                np.float32(inv_height),
                                np.float32(self.theta_scale),
                                np.float32(self.u_scale),
                                np.float32(self.v_scale),
                                np.float32(u_offset),
                                np.float32(v_offset),
                                self.chromakey_array,
                                block=block_size,
                                grid=grid_size,
                                stream=self.cuda_stream
                            )
                        else:
                            self.render_kernel(
                                np.intp(src_image.data.ptr),
                                np.int32(src_image.strides[0]),
                                np.int32(input_width),
                                np.int32(input_height),
                                np.uint64(surf_obj.value),
                                np.int32(output_width),
                                np.int32(output_height),
                                rot_array,
                                np.float32(tan_left),
                                np.float32(tan_down),
                                np.float32(tan_width),
                                np.float32(tan_height),
                                np.float32(inv_width),
                                np.float32(inv_height),
                                np.float32(self.theta_scale),
                                np.float32(self.u_scale),
                                np.float32(self.v_scale),
                                np.float32(u_offset),
                                np.float32(v_offset),
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

    #
    # Stub to provide a basic invoke() interface for non-VR aware callers.
    #
    identity_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    class Fov(object):
        def __init__(self, width, height, angle_up=36):
            import math
            aspect_ratio    = width / height
            self.angle_down = self.angle_up    = math.radians(angle_up)
            self.angle_left = self.angle_right = math.atan(aspect_ratio * math.tan(self.angle_up))
            self.angle_left *= -1
            self.angle_up   *= -1

    def invoke(self, source_image, dest_image, dest_size, timers=None):
        "Returns straight ahead view of left eye by default."

        if self.fov is None:
            self.fov = self.Fov(*dest_size, 50)  # Wide angle FOV more useful?

        self.invoke_vr(source_image, dest_image, dest_size, self.identity_matrix, self.fov)

    def project_ray_to_screen(self, rot_matrix, leveling_offset=0, yaw_offset=0, screen_width=4.4, screen_height=2.475, screen_distance=3.0):
        """Project a ray (from rot_matrix forward direction) onto the virtual screen.
        
        Args:
            rot_matrix: 3x3 rotation matrix representing the ray direction (e.g., from controller aim pose)
            All parameters are as in invoke_vr() above.
        
        Returns:
            (u, v) in [0, 1] range if ray hits screen, None otherwise.
            (0, 0) is top-left, (1, 1) is bottom-right.
        """
        if self.projection not in ('flat', 'mono'): # Unlikely we'll ever need more but maybe.
            return None
        
        # Apply the same transformations as invoke_vr does to the scene
        rot_matrix = self.apply_view_adjustments(rot_matrix, leveling_offset, yaw_offset)
        
        # Extract forward direction from rotation matrix (third column, negated)
        # In OpenXR convention, forward is -Z
        ray = -rot_matrix[:, 2]
        
        # Intersect ray with screen plane at -screen_distance along Z axis
        # Screen is perpendicular to -Z, centered at origin
        if ray[2] >= 0:
            # Ray pointing away from screen
            return None
        
        # Compute t where ray intersects the plane z = -screen_distance
        # ray starts at origin (position ignored), so: t * ray.z = -screen_distance
        t = -screen_distance / ray[2]
        
        # Compute hit point
        hit_x = ray[0] * t
        hit_y = ray[1] * t
        
        # Compute screen dimensions
        half_width  = screen_width  * 0.5
        half_height = screen_height * 0.5
        
        # Check if within bounds
        if abs(hit_x) > half_width or abs(hit_y) > half_height:
            return None
        
        # Convert to UV (0=top-left, 1=bottom-right)
        u = (hit_x / screen_width) + 0.5
        v = 0.5 - (hit_y / screen_height)  # Y is inverted (up is positive in 3D)
        
        return (u, v)

