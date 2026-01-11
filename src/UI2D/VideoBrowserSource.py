
#
# Thin wrapper that converts our generic VideoBrowserPlugin into a VideoSource
#
# We could modularize all this better later if we end up with more plugins of this nature.
#

from VideoSource import VideoSource
from .Renderer   import DualRenderer

class VideoBrowserSource(VideoSource):

    #
    # This wrapper class is to map the Renderer style plugin (which is
    #   very generic and works outside the setting of a video player..)
    #   to a "video source" which is an API really optimized for video
    #   playback but which we've usurped for UI too.
    #
    # (NOTE for instance that if we wanted to use a DualRenderer style
    #  plugin as an Overlay during active video playing, we might
    #  do so directly and not as a VideoSource since there's no need
    #  for this faux "get_frame" API.  This class here is really just
    #  duct tape to cram a DualRenderer into the existing video playback
    #  pipeline...)
    #
    def __init__(self, rootdir, size, create=False):

        self.size      = (self.width, self.height) = size
        self.framerate = 60.0
        self.renderer  = VideoBrowserRenderer(rootdir, size, create=create)

    def get_frame(self, frame_number, io):
        if io.size != self.size:
            #print(f"DEBUG: VideoBroswerSource detected size change -> {io.size}")
            self.size = (self.width, self.height) = io.size
        if io.play is not None:
            return None # This ends this source stream with a single frame delay, once user selects video to play.
        return self.renderer.render_cupy(io.size, io)

    def handle_events(self, io):
        # This is a gross reach-in hack to make escape act like the Back button
        browser = self.renderer.browser
        if 'escape' in io.keystrokes:
            if browser.go_back():
                io.keystrokes.remove('escape')

    def close(self):
        self.renderer.close()

class VideoBrowserRenderer(DualRenderer):

    #
    # This wrapper class is to map render_gl to render_cupy -- i.e.,
    #  to get a cupy image of the VideoBrowserPlugin's imgui
    #

    def __init__(self, rootdir, size, **kargs):
        DualRenderer.__init__(self)

        from .VideoBrowser import VideoBrowserPlugin
        self.browser = VideoBrowserPlugin(rootdir, size, **kargs)

    def render_gl(self, size, io):
        self.browser.render(io)

