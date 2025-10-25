import numpy as np
from OpenGL.GL import *
from OpenGL import GLX
import xr
import ctypes


class OpenXRDevice(object):
    """OpenXR VR headset output device.

    ALLOCATION:
      - glfw must be initialized and with an active window (even if hidden) before creating this object.
      - OpenXR runtime creates swapchain images
      - We expose them for CUDA registration

    Init chain:
      - Call init() to create OpenXR session and swapchains
      - Call get_swapchain_images() to get GL textures for CUDA registration
      - Downstream mappers register these with CUDA

    Runtime:
      - begin_frame() -> acquires swapchain images, returns view info
      - end_frame()   -> releases images and submits to compositor
    """

    def __init__(self, app_name="VR Video Player"):
        self.app_name = app_name

        # OpenXR handles
        self.instance          = None
        self.system_id         = None
        self.session           = None
        self.space             = None
        self.swapchains        = {}  # 'left' and 'right'
        self.swapchain_images  = {}  # GL texture handles

        # View configuration
        self.view_config_type  = xr.ViewConfigurationType.PRIMARY_STEREO
        self.views             = None
        self.view_configs      = None

        # Frame state
        self.frame_state       = None
        self.swapchain_indices = {}
        self.images_acquired   = False  # Track if images are acquired

        # Session state
        self.session_running   = False
        self.session_state     = None

    def init(self):
        """
        Initialize OpenXR session and create swapchains.
        Returns dict of swapchain dimensions: {'width': int, 'height': int}
        """
        if False:
            # This is handled outside of this module now.
            import glfw
            glfw.init()
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            window = glfw.create_window(640, 480, "Hidden", None, None)
            glfw.make_context_current(window)

        # Create instance
        try:
            self.instance = xr.create_instance(
                xr.InstanceCreateInfo(
                    application_info=xr.ApplicationInfo(
                        application_name=self.app_name,
                        application_version=1,
                        engine_name="CustomPlayer",
                        engine_version=1,
                        api_version=xr.XR_CURRENT_API_VERSION
                    ),
                    enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME]
                )
            )
        except Exception as e:
            raise Exception(f"OpenXR couldn't open the XR device.  ({e})")

        # Get system
        self.system_id = xr.get_system(
            self.instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        )

        # Get view configuration
        self.view_configs = xr.enumerate_view_configuration_views(
            self.instance,
            self.system_id,
            self.view_config_type
        )

        print("View Configuration:")
        for i, view in enumerate(self.view_configs):
            print(f"   Eye {i}: {view.recommended_image_rect_width}x{view.recommended_image_rect_height}")

        # Call OpenGL graphics requirements
        proc_addr = xr.get_instance_proc_addr(self.instance, "xrGetOpenGLGraphicsRequirementsKHR")
        graphics_requirements = xr.GraphicsRequirementsOpenGLKHR()
        func = ctypes.cast(proc_addr, ctypes.CFUNCTYPE(
            ctypes.c_int, xr.Instance, xr.SystemId,
            ctypes.POINTER(xr.GraphicsRequirementsOpenGLKHR)
        ))
        func(self.instance, self.system_id, ctypes.byref(graphics_requirements))

        print(f"OpenGL Requirements: min={graphics_requirements.min_api_version_supported}, max={graphics_requirements.max_api_version_supported}")

        # Create OpenGL graphics binding
        # This assumes GL context is already created and current
        display  = GLX.glXGetCurrentDisplay()
        context  = GLX.glXGetCurrentContext()
        drawable = GLX.glXGetCurrentDrawable()

        print(f"Display: {display}")
        print(f"Context: {context}")
        print(f"Drawable: {drawable}")

        graphics_binding = xr.GraphicsBindingOpenGLXlibKHR(
            x_display    = display,
            glx_context  = context,
            glx_drawable = drawable
        )

        # Create session
        self.session = xr.create_session(
            self.instance,
            xr.SessionCreateInfo(
                system_id=self.system_id,
                next=ctypes.cast(ctypes.pointer(graphics_binding), ctypes.c_void_p)
            )
        )

        # Create reference space
        self.space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.STAGE,
                pose_in_reference_space=xr.Posef()
            )
        )

        # Create swapchains for each eye
        for i, view_config in enumerate(self.view_configs):
            eye = 'left' if i == 0 else 'right'

            swapchain = xr.create_swapchain(
                self.session,
                xr.SwapchainCreateInfo(
                    usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT,
                    #format=GL_RGBA8,       # Linear
                    format=GL_SRGB8_ALPHA8, # Gamma compressed ; expected by the back end.
                    sample_count=1,
                    width=view_config.recommended_image_rect_width,
                    height=view_config.recommended_image_rect_height,
                    face_count=1,
                    array_size=1,
                    mip_count=1
                )
            )

            self.swapchains[eye] = swapchain

            # Get swapchain images (GL textures created by runtime)
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self.swapchain_images[eye] = [img.image for img in images]

        # Begin session
        # NOTE: We don't call begin_session here anymore - we wait for READY event
        # xr.begin_session(
        #     self.session,
        #     xr.SessionBeginInfo(
        #         primary_view_configuration_type=self.view_config_type
        #     )
        # )
        print("DEBUG: Session created, waiting for READY state event...")

        # Return details for downstream configuration
        view_config = self.view_configs[0]  # Both eyes same resolution
        return {
            'size': (view_config.recommended_image_rect_width, view_config.recommended_image_rect_height),
            'swapchain_images': self.swapchain_images,
        }

    def get_swapchain_images(self):
        """
        Get swapchain images for CUDA registration.
        Returns: {'left': [gl_textures...], 'right': [gl_textures...]}
        """
        return self.swapchain_images

    def poll_events(self):
        """
        Poll for OpenXR events and handle session state changes.
        Should be called each frame before begin_frame().
        Returns: True if session is still active, False if should exit
        """
        while True:
            try:
                # poll_event returns just the event buffer
                event_buffer = xr.poll_event(self.instance)

                event_type = event_buffer.type

                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    event = ctypes.cast(
                        ctypes.byref(event_buffer),
                        ctypes.POINTER(xr.EventDataSessionStateChanged)
                    ).contents

                    self.session_state = event.state
                    print(f"DEBUG: Session state changed to {self.session_state}")

                    if event.state == xr.SessionState.READY:
                        print("DEBUG: Session READY - beginning session")
                        xr.begin_session(
                            self.session,
                            xr.SessionBeginInfo(
                                primary_view_configuration_type=self.view_config_type
                            )
                        )
                        self.session_running = True

                    elif event.state == xr.SessionState.STOPPING:
                        print("DEBUG: Session STOPPING - ending session")
                        xr.end_session(self.session)
                        self.session_running = False

                    elif event.state == xr.SessionState.EXITING:
                        print("DEBUG: Session EXITING")
                        return False

                    elif event.state == xr.SessionState.LOSS_PENDING:
                        print("DEBUG: Session LOSS_PENDING")
                        return False

                elif event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                    print("DEBUG: Instance loss pending")
                    return False

            except xr.exception.EventUnavailable:
                # No more events available - this is normal
                break
            except Exception as e:
                print(f"DEBUG: Unexpected error polling events: {e}")
                import traceback
                traceback.print_exc()
                break

        return True

    def begin_frame(self):
        """
        Begin frame and acquire swapchain images.
        Returns: {
            'should_render': bool,
            'predicted_display_time': int,
            'left_index': int,
            'right_index': int,
            'views': [view poses...]
        }
        """
        # Don't call wait_frame if session isn't running
        if not self.session_running:
            return {'should_render': False}

        # Wait for next frame
        self.frame_state = xr.wait_frame(self.session, xr.FrameWaitInfo())

        # Begin frame
        xr.begin_frame(self.session, xr.FrameBeginInfo())

        # Reset acquired flag
        self.images_acquired = False

        if not self.frame_state.should_render:
            print("DEBUG: should_render is False")
            return {'should_render': False}

        # Locate views
        view_locate_info = xr.ViewLocateInfo(
            view_configuration_type=self.view_config_type,
            display_time=self.frame_state.predicted_display_time,
            space=self.space
        )
        view_state, views = xr.locate_views(self.session, view_locate_info)
        self.views = views

        #print(f"DEBUG: Located {len(views)} views, acquiring swapchains...")

        # Acquire swapchain images
        for eye in ['left', 'right']:
            acquire_info = xr.SwapchainImageAcquireInfo()
            index = xr.acquire_swapchain_image(self.swapchains[eye], acquire_info)

            # Wait for image to be ready
            wait_info = xr.SwapchainImageWaitInfo(timeout=xr.INFINITE_DURATION)
            xr.wait_swapchain_image(self.swapchains[eye], wait_info)

            self.swapchain_indices[eye] = index
            #print(f"DEBUG: {eye} eye index: {index}")

        # Mark images as acquired
        self.images_acquired = True

        return {
            'should_render': True,
            'predicted_display_time': self.frame_state.predicted_display_time,
            'left_index': self.swapchain_indices['left'],
            'right_index': self.swapchain_indices['right'],
            'views': views
        }

    def end_frame(self):
        """
        Release swapchain images and submit frame to compositor.
        """
        # Don't call end_frame if session isn't running
        if not self.session_running:
            return

        # Only release swapchain images if they were acquired
        if self.images_acquired:
            #print("DEBUG: Releasing swapchain images")
            for eye in ['left', 'right']:
                xr.release_swapchain_image(
                    self.swapchains[eye],
                    xr.SwapchainImageReleaseInfo()
                )
            self.images_acquired = False

        # Submit layers to compositor
        if self.frame_state.should_render:
            #print("DEBUG: Building projection layer")
            projection_views = []
            for i, view in enumerate(self.views):
                eye = 'left' if i == 0 else 'right'
                view_config = self.view_configs[i]

                projection_views.append(
                    xr.CompositionLayerProjectionView(
                        pose=view.pose,
                        fov=view.fov,
                        sub_image=xr.SwapchainSubImage(
                            swapchain=self.swapchains[eye],
                            image_rect=xr.Rect2Di(
                                offset=xr.Offset2Di(0, 0),
                                extent=xr.Extent2Di(
                                    view_config.recommended_image_rect_width,
                                    view_config.recommended_image_rect_height
                                )
                            )
                        )
                    )
                )

            # Convert projection_views to a ctypes array
            ProjectionViewArray    = xr.CompositionLayerProjectionView * len(projection_views)
            projection_views_array = ProjectionViewArray(*projection_views)

            projection_layer = xr.CompositionLayerProjection(
                next       = None,                          # What does this do?  Is it needed?
                space      = self.space,
                view_count = len(projection_views),
                views      = projection_views_array
            )

            if True:
                base_layer = ctypes.cast(ctypes.byref(projection_layer), ctypes.POINTER(xr.CompositionLayerBaseHeader)).contents
                layers = [ctypes.pointer(base_layer)]  # Use pointer to base_layer
            else:
                layers = [projection_layer]

            #print(f"DEBUG: Submitting {len(layers)} layers")
        else:
            layers = []
            #print("DEBUG: Submitting 0 layers (should_render=False)")

        # End frame
        xr.end_frame(
            self.session,
            xr.FrameEndInfo(
                display_time=self.frame_state.predicted_display_time,
                environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                layer_count = len(layers),  # Is this right?
                layers=layers
            )
        )
        #print("DEBUG: Frame ended")

    def close(self):
        """Clean up OpenXR resources."""
        if self.space:
            xr.destroy_space(self.space)

        for swapchain in self.swapchains.values():
            xr.destroy_swapchain(swapchain)

        if self.session:
            # Only end session if it's running
            if self.session_running:
                try:
                    xr.end_session(self.session)
                except:
                    pass  # Session may already be ended
            xr.destroy_session(self.session)

        if self.instance:
            xr.destroy_instance(self.instance)

