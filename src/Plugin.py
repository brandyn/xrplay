
#
# All plugins should inherit from this class so the provide all the
#  expected methods.  Only those relevant to the plugin need to be
#  implemented, with these stubs being safe noops.
#

class Plugin(object):

    def init_video(self, video_path, info, fps):
        """Called once when a new video source is started.
        (Currently not called for the browser.)

        info is a dict from Info.get() containing persistent information
            specific to the particular video.  You can read/write it at
            will (limited to picklable values) and it will be saved later
            if changed.
        """
        pass

    def source_frame(self, frame, paused, speed, skipped, frame_number):
        """This allows the plugin to read and potentially modify the frame.
        This is called right after the original source (e.g. video decoder)
        produces the frame.

        frame        - a cupy array (height, width, 3)
        paused       - True if the video is currently paused
        speed        - the current playback speed as a float ratio, usually 1.0
        skipped      - how many frames we advanced since the last call (normally 1 ; can be negative!)
        frame_number - just that

        You must return the resulting frame, which can be a new frame, or
            potentially modified in place.  Technically resizing _is_ allowed
            here, but be consistent because downstream reconfigures significantly
            any time the frame size changes.
        """
        return frame

    def ui(self, paused, io, left=None):
        """This is called, always later than source_frame(), to allow the
            plugin to do any UI.

        paused - True if the video is paused.
        left   - the controller dict for the left OpenXR controller, if available.
        """
        #   left: {
        #          'grip_pose': {'orientation': (0.5574334263801575,
        #                                        -0.12656213343143463,
        #                                        -0.31022343039512634,
        #                                        0.7596127390861511),
        #                        'position': (-0.0639796257019043,
        #                                     0.7555080652236938,
        #                                     -0.11990729719400406)},
        #          'squeeze': 1.0,
        #          'thumbstick': (-0.43317973613739014, -0.9013031721115112),
        #          'trigger': 1.0,
        #          'x_button': 1,
        #          'y_button': 1},
        pass

    def close_video(self, video_path, info):
        """This is called when a video is closed (note a video may LOOP many
            times before being closed).  It should be always paired 1:1 with
            init_video() but to be save close_video() should be idempotent.
        """
        pass

    def close(self):
        """Final cleanup before shutting down the plugin entirely.
        No other calls will come after this.
        """
        pass

    # == VideoBrowser (imgui version) related ==

    def browser_render_filters(self):
        """This is called in the imgui rendering sequence right after
            the star ratings and such, and before tags.

        You can add imgui elements here as desired.

        Return True iff the resulting filter state has changed.
        """
        return False

    def browser_filter(self, video):
        """Return True if video passes current filters.

        video here is a dict with various video attributes.  See VideoBrowser.py
        """
        return True

