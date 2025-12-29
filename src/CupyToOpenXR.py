
from OpenGL import GL

import pycuda.gl as cuda_gl
import numpy     as np

class CupyToOpenXR:
    """Renders a cupy source image to an OpenXR device in one GPU pass.

    Init chain:
      - cuda and GL contexts must already be active
      - Pre-registers two output swapchains of GL textures (typically from OpenXRDevice), as:
            {'left': [gl_textures...], 'right': [gl_textures...]}
    
    Runtime:
      - cuda and GL contexts must already be active
      - invoke(image, left_index, right_index, views)
            where *_index says which output image in each swapchain to use.
    """
    def __init__(self, projector, swapchain_images=None):
        """projector is a CupyToGLTexture mapper supporting invoke_vr()
        swapchain_images must be passed here or provided later to .init()
        """
        self.projector       = projector
        self.cuda_swapchains = {'left': [], 'right': []}

        if swapchain_images is not None:
            self.init(swapchain_images)

    def init(self, swapchain_images):
        """Register all swapchain images with CUDA.
        
        Args:
            swapchain_images: {'left': [gl_textures], 'right': [gl_textures]}
        
        Returns:
            None (terminal mapper)

        Don't call this if you already provided the swapchains to the constructor.
        """
        for eye in ['left', 'right']:
            for gl_texture in swapchain_images[eye]:
                registered = self.projector.register_texture(gl_texture)
                self.cuda_swapchains[eye].append(registered)
    
    def invoke(self, image, left_index, right_index, views=None, leveling_offset=0, yaw_offset=0):
        """
        Render from cuda memory straight to output GL texture (swapchain image)
            with the currently selected kernel (e.g., 180 SBS), using view
            poses (usually OpenXR Views) corresponding to each output where applicable.
        
        Args:
            image: cupy RGB input data
            left_index: Left eye swapchain image index
            right_index: Right eye swapchain image index
            views: render orientation for each view --
                    list of 2 view (usually xr.View) objects with .pose and .fov
            leveling_offset: radians of rotational pitch adjustment.  Positive tilts scene down.
        """
        if views is None or len(views) < 2:
            raise ValueError("VR rendering requires 2 views")
        
        for eye_idx, eye_name, swapchain_index, view in [
                                (0,  'left',  left_index, views[0]), 
                                (1, 'right', right_index, views[1])]:
            # Extract rotation matrix from quaternion and convert to rotation matrix (3x3)
            quat       = view.pose.orientation
            rot_matrix = self._quat_to_matrix(quat.x, quat.y, quat.z, quat.w)
            self.projector.invoke_vr(
                    src_image       = image,
                    dst_image       = self.cuda_swapchains[eye_name][swapchain_index],
                    rot_matrix      = rot_matrix,
                    fov             = view.fov,
                    leveling_offset = leveling_offset,
                    yaw_offset      = yaw_offset,
                    eye             = eye_idx
                )

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
                self.projector.unregister_texture(registered)

