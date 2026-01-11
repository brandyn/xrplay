"""
Video Browser UI - ImGui-based file browser with tagging and rating.

Usage:
    browser = VideoBrowserPlugin(video_root="/path/to/videos")
    browser.render(io)  # Render's the next frame and returns it...
"""

import imgui
from imgui.integrations.opengl import ProgrammablePipelineRenderer
from OpenGL.GL import *
from pathlib import Path
import numpy as np
import sqlite3
import math
import xxhash

# From codes in IO.py to imgui codes:
if True:    # Can't do it this way if using a back end integration that will overwrite imgui's key_map...
    key_map = {
        'control': imgui.KEY_MOD_CTRL,
        'shift': imgui.KEY_MOD_SHIFT,
        # Arrows
        'right': imgui.KEY_RIGHT_ARROW,
        'left': imgui.KEY_LEFT_ARROW,
        'up': imgui.KEY_UP_ARROW,
        'down': imgui.KEY_DOWN_ARROW,
        # Etc
        'home': imgui.KEY_HOME,
        'page-up': imgui.KEY_PAGE_UP,
        'page-down': imgui.KEY_PAGE_DOWN,
        'end': imgui.KEY_END,
        'escape': imgui.KEY_ESCAPE,
        # Single letters (for shortcuts like Ctrl+C, Ctrl+V, etc.)
        'a': imgui.KEY_A,
        'c': imgui.KEY_C,
        'v': imgui.KEY_V,
        'x': imgui.KEY_X,
        'y': imgui.KEY_Y,
        'z': imgui.KEY_Z,
        # Additional useful keys
        'backspace': imgui.KEY_BACKSPACE,
        'delete': imgui.KEY_DELETE,
        'enter': imgui.KEY_ENTER,
        'tab': imgui.KEY_TAB,
        ' ': imgui.KEY_SPACE,
    }
else:   # E.g., if you are using the glfw integration, you'll have to do this (and comment out the io.key_map code below):
    import glfw
    key_map = {
        # Modifiers (ImGui uses separate flags, but for keys_down in legacy mode, use left variants; add right if needed)
        'control': glfw.KEY_LEFT_CONTROL,   # or glfw.KEY_RIGHT_CONTROL
        'shift': glfw.KEY_LEFT_SHIFT,       # or glfw.KEY_RIGHT_SHIFT

        # Arrows
        'right': glfw.KEY_RIGHT,
        'left': glfw.KEY_LEFT,
        'up': glfw.KEY_UP,
        'down': glfw.KEY_DOWN,

        # Navigation keys
        'home': glfw.KEY_HOME,
        'page-up': glfw.KEY_PAGE_UP,
        'page-down': glfw.KEY_PAGE_DOWN,
        'end': glfw.KEY_END,
        'escape': glfw.KEY_ESCAPE,

        # Single letters for shortcuts (e.g., Ctrl+C, Ctrl+V)
        # In GLFW, 'A' to 'Z' are ASCII values: 'A' = 65, etc.
        'a': glfw.KEY_A,
        'c': glfw.KEY_C,
        'v': glfw.KEY_V,
        'x': glfw.KEY_X,
        'y': glfw.KEY_Y,
        'z': glfw.KEY_Z,

        # Additional useful keys
        'backspace': glfw.KEY_BACKSPACE,
        'delete': glfw.KEY_DELETE,
        'enter': glfw.KEY_ENTER,
        'tab': glfw.KEY_TAB,
        ' ': glfw.KEY_SPACE,
    }

class VideoBrowserPlugin(object):
    """
    Video browser with tags, ratings, and filtering.
    Pure UI component - no VR dependencies.
    """
    ALLOWED_EXTENSIONS = {'.mp4', '.mkv', '.webm', '.mov', '.avi', '.flv', '.wmv', '.m4v'}

    last_selected_video = None  # HACK to persist a little state across instantiations...

    def __init__(self, video_root, size=(1920, 1080), readonly=False, create=False):
        """readonly blocks any attempts to change the database (or delete
            files) and simply logs them instead.
        """
        self.video_root = Path(video_root)
        self.size       = size
        self.readonly   = readonly
        
        # UI Scale - can be changed at runtime
        self.ui_scale = 2.0
        
        # Database
        self.db_path   = self.db_path = self.video_root / "xrplay.db"
        self.db_conn   = None
        self.trash_dir = self.video_root / "TRASH"    # For now...
        
        # ImGui context
        self.context = imgui.create_context()
        imgui.set_current_context(self.context)
        
        iio               = imgui.get_io()
        iio.display_size  = size
        iio.ini_file_name = "".encode()      # Disable creation of ./imgui.ini
        iio.mouse_draw_cursor = True        # Probably need a flag for this since we don't always want it...
        
        # Style
        imgui.style_colors_dark()
        self._apply_style_scale()
        
        # Renderer - initialized on first render
        self.renderer = None
        
        # FBO for offscreen rendering - just need one FBO and depth buffer
        self.fbo       = None
        self.depth_rbo = None
        
        # UI State
        self.selected_tags     = set()
        self.min_rating        = 0
        self.filter_no_tags    = False
        self.filter_no_rating  = False
        self.sort_by           = "name"  # name, date, rating, size
        self.search_text       = ""
        
        # File list cache
        self.all_videos      = []
        self.filtered_videos = []
        self.all_tags        = []
        
        # Selection state
        self.selected_video = VideoBrowserPlugin.last_selected_video
        self.scroll_to_hash = None  # For "back" navigation
        
        # Action callback
        self.action_callback = None
        
        # Text input buffers
        self.search_buffer  = ""
        self.new_tag_buffer = ""
        
        # Grid layout
        self.min_column_width = 200  # Minimum width per grid item
        
        # Pre-calculate star geometry (unit size, will be scaled)
        self._star_points = []
        for i in range(10):
            angle = (i * 36 - 90) * 3.14159 / 180.0
            radius = (0.5 if i % 2 == 0 else 0.2) * 1.5 # 1.5 just a fudge factor to make them a little bigger
            self._star_points.append((math.cos(angle) * radius, math.sin(angle) * radius + 0.3))    # 0.3 = alignment fudge

        # Key-down difference tracking:
        self.keys_down = set()

        # Initialize
        self._init_database(create=create)
        self._refresh_file_list()
    
        # Identity key map because we're using imgui's own key codes.  This will clash with
        #  any imgui integrations (glfw, sdl2, etc) so beware...
        key_map = imgui.get_io().key_map
        for i in range(len(key_map)):
            key_map[i] = i

        # Thumbnail cache
        self.thumbnail_cache    = {}    # hash -> OpenGL texture ID
        self.thumbnail_queue    = []    # LRU queue of hashes
        self.max_cached_thumbs  = 100   # Keep N most recent
        self.thumbnails_loading = set() # Currently loading hashes
        self.thumbnails_missing = set() # Known missing thumbnails (don't check again)
        
        # For async loading
        import threading
        self.thumbnail_load_queue    = []  # [(hash, jpg_path), ...]
        self.thumbnail_load_lock     = threading.Lock()
        self.thumbnail_loader_thread = None
        self._loaded_thumbnails      = []   # Transient from bg thread to main thread?

    def set_ui_scale(self, scale=1.0):
        """Set UI scale factor (affects all UI elements and fonts)."""
        self.ui_scale = max(0.5, min(scale*2.0, 4.0))  # Clamp to reasonable range
        self._apply_style_scale()
    
    def _apply_style_scale(self):
        """Apply current UI scale to ImGui style."""
        style = imgui.get_style()
        
        # Base style values (at scale 1.0)
        base_window_rounding  = 8.0
        base_frame_rounding   = 4.0
        base_scrollbar_size   = 20.0
        base_frame_padding    = (4.0, 3.0)
        base_item_spacing     = (8.0, 4.0)
        base_item_inner_spacing = (4.0, 4.0)
        
        # Apply scale
        style.window_rounding     = base_window_rounding * self.ui_scale
        style.frame_rounding      = base_frame_rounding * self.ui_scale
        style.scrollbar_size      = base_scrollbar_size * self.ui_scale
        style.frame_padding       = (base_frame_padding[0] * self.ui_scale,
                                     base_frame_padding[1] * self.ui_scale)
        style.item_spacing        = (base_item_spacing[0] * self.ui_scale,
                                     base_item_spacing[1] * self.ui_scale)
        style.item_inner_spacing  = (base_item_inner_spacing[0] * self.ui_scale,
                                     base_item_inner_spacing[1] * self.ui_scale)
        
        # Font scale
        iio = imgui.get_io()
        iio.font_global_scale = self.ui_scale
    
    def _scaled(self, value):
        """Scale a single value by UI scale."""
        return value * self.ui_scale
    
    def _refresh_file_list(self):
        """Reload file list from database and apply filters."""
        if not self.db_conn:
            self.all_videos      = []
            self.filtered_videos = []
            self.all_tags        = []
            return
        
        # Load all videos
        rows = self.db_conn.execute(
            'SELECT hash, file_path, file_size, tags, rating FROM file_info'
        ).fetchall()
        
        self.all_videos = []
        for row in rows:
            tags = row['tags'].split('|') if row['tags'] else []
            path = Path(row['file_path'])
            
            self.all_videos.append({
                'hash'     : row['hash'],
                'name'     : path.name,
                'stem'     : path.stem,
                'path'     : row['file_path'],
                'size'     : row['file_size'],
                'tags'     : tags,
                'path_tags': list(path.parent.parts) if path.parent != Path('.') else [],
                'rating'   : row['rating'] or 0,
            })
        
        # Load all tags
        tag_counts = {}
        for video in self.all_videos:
            for tag in video['tags'] + video['path_tags']:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        self.all_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Apply filters
        self._apply_filters()
    
    def _apply_filters(self):
        """Filter and sort video list."""
        filtered = list(self.all_videos)
        
        # Search filter
        if self.search_text:
            search_lower = self.search_text.lower()
            filtered = [v for v in filtered if search_lower in v['name'].lower()]
        
        # Tag filters
        if self.filter_no_tags:
            filtered = [v for v in filtered if not v['tags']]
        elif self.selected_tags:
            filtered = [
                v for v in filtered
                if all(tag in v['tags'] or tag in v['path_tags'] 
                      for tag in self.selected_tags)
            ]
        
        # Rating filter
        if self.filter_no_rating:
            filtered = [v for v in filtered if v['rating'] == 0]
        else:
            filtered = [v for v in filtered if v['rating'] >= self.min_rating]
        
        # Sort
        if self.sort_by == "name":
            filtered.sort(key=lambda v: v['name'].lower())
        elif self.sort_by == "date":
            # Note: we don't have mod_time in our quick query, would need to add
            filtered.sort(key=lambda v: v['name'].lower())
        elif self.sort_by == "rating":
            filtered.sort(key=lambda v: v['rating'], reverse=True)
        elif self.sort_by == "size":
            filtered.sort(key=lambda v: v['size'], reverse=True)
        
        self.filtered_videos = filtered
    
    def _format_size(self, bytes_val):
        """Format file size."""
        gb = bytes_val / (1024**3)
        if gb >= 1:
            return f'{gb:.2f} GB'
        mb = bytes_val / (1024**2)
        return f'{mb:.1f} MB'
    
    def _draw_star(self, draw_list, pos_x, pos_y, size, filled, color):
        """Draw a star shape at given position using pre-calculated geometry."""
        # Convert color tuple to packed int (ABGR)
        r, g, b, a = color
        packed_color = (int(a * 255) << 24) | (int(b * 255) << 16) | (int(g * 255) << 8) | int(r * 255)
        
        if filled:
            # Use triangles to fill the star (draw from center)
            for i in range(10):
                x1 = pos_x + self._star_points[i][0] * size
                y1 = pos_y + self._star_points[i][1] * size
                x2 = pos_x + self._star_points[(i + 1) % 10][0] * size
                y2 = pos_y + self._star_points[(i + 1) % 10][1] * size
                draw_list.add_triangle_filled(
                    pos_x, pos_y,  # Center
                    x1, y1,         # Current point
                    x2, y2,         # Next point
                    packed_color
                )
        else:
            # Draw outline
            for i in range(10):
                x1 = pos_x + self._star_points[i][0] * size
                y1 = pos_y + self._star_points[i][1] * size
                x2 = pos_x + self._star_points[(i + 1) % 10][0] * size
                y2 = pos_y + self._star_points[(i + 1) % 10][1] * size
                draw_list.add_line(x1, y1, x2, y2, packed_color, 1.0)
    
    def _draw_rating_stars(self, rating, max_rating=5, size=None):
        """Draw rating stars and return width used."""
        if size is None:
            size = self._scaled(12)
        
        draw_list = imgui.get_window_draw_list()
        cursor_pos = imgui.get_cursor_screen_pos()
        
        filled_color = (0.9, 0.7, 0.2, 1.0)
        empty_color  = (0.4, 0.4, 0.4, 1.0)
        
        spacing = size * 2.2
        for i in range(max_rating):
            star_x = cursor_pos[0] + i * spacing
            star_y = cursor_pos[1] + size * 0.5
            filled = (i < rating)
            self._draw_star(draw_list, star_x, star_y, size, filled,
                           filled_color if filled else empty_color)
        
        total_width = max_rating * spacing
        imgui.dummy(total_width, size)
        return total_width
    
    def _trigger_action(self, action_type, data):
        """Trigger action callback."""
        if self.action_callback:
            self.action_callback(action_type, data)
    
    def set_action_callback(self, callback):
        """Set callback for actions. Signature: callback(action_type, data)"""
        self.action_callback = callback
    
    def render(self, io):
        """Render UI to current GL FBO.
        
        io.:
            size = (width, height): Render target dimensions
            mouse_x, mouse_y: Mouse position in pixels
            mouse_down: Boolean, is left mouse button down
            mouse_scroll: Mouse scrolling delta
            text_input: String of characters typed this frame (e.g., "a", "hello", "")
            keys_down: set, list or tuple of currently pressed keys, as specified in IO.py

            play: set by this method to relative path to the video to play, if user selects.
        """
        # Initialize GL resources on first render
        if self.renderer is None:
            self.renderer = ProgrammablePipelineRenderer()
        
        # Upload any newly loaded thumbnails to GPU (must be on main thread)
        with self.thumbnail_load_lock:
            if self._loaded_thumbnails:
                to_upload = self._loaded_thumbnails[:]
                self._loaded_thumbnails.clear()
            else:
                to_upload = []
        for file_hash, img_bytes, img_width, img_height in to_upload:
            try:
                # Create OpenGL texture
                texture_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
                
                glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0,
                            GL_RGB, GL_UNSIGNED_BYTE, img_bytes)
                
                glBindTexture(GL_TEXTURE_2D, 0)  # Unbind
                
                # Add to cache
                self.thumbnail_cache[file_hash] = texture_id
                self.thumbnail_queue.append(file_hash)
                self.thumbnails_loading.discard(file_hash)
                
                # Evict old thumbnails if cache too large
                while len(self.thumbnail_queue) > self.max_cached_thumbs:
                    old_hash = self.thumbnail_queue.pop(0)
                    if old_hash in self.thumbnail_cache:
                        glDeleteTextures([self.thumbnail_cache[old_hash]])
                        del self.thumbnail_cache[old_hash]
            except Exception as e:
                print(f"Failed to create texture for {file_hash}: {e}")
                self.thumbnails_loading.discard(file_hash)


        # Update IO state
        iio = imgui.get_io()

        if io.size != self.size: # Handle resize
            #print(f"DEBUG: Browser detected size change -> {io.size}")
            self.size        = io.size  # Tell anyone who cares what size we'll be rendering.
            iio.display_size = io.size  # Tell imgui what size to render to

        iio.mouse_pos      = (io.mouse_x, io.mouse_y)
        iio.mouse_down[0]  = io.mouse_down
        iio.mouse_wheel    = io.mouse_scroll_y
        iio.delta_time     = 1.0 / 60.0 # TODO -- we can do better, but does it matter?
        
        # Add text input characters
        if io.text_input:
            for char in io.text_input:
                iio.add_input_character(ord(char))
        
        # Update special key states
        if io.keys_down or self.keys_down:
            keys_down      = set(io.keys_down)
            new_keys_down  = keys_down - self.keys_down
            old_keys_down  = self.keys_down - keys_down
            self.keys_down = keys_down
            for key in new_keys_down:
                if key in key_map:
                    iio.keys_down[key_map[key]] = True
            for key in old_keys_down:
                if key in key_map:
                    iio.keys_down[key_map[key]] = False
        
        #
        # Now we do the actual rendering:
        #
        glViewport(0, 0, *io.size)
        glClearColor(0.1, 0.1, 0.12, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Render UI
        imgui.new_frame()
        
        if self.selected_video:
            if self._render_detail_view():
                io.play = self.selected_video['path']
        else:
            self._render_list_view()
        
        imgui.render()
        self.renderer.render(imgui.get_draw_data())

    def _render_list_view(self):
        """Render main list view with filters."""
        # Full window
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*self.size)
        
        imgui.begin("Video Browser",
                   flags=imgui.WINDOW_NO_TITLE_BAR |
                         imgui.WINDOW_NO_RESIZE |
                         imgui.WINDOW_NO_MOVE |
                         imgui.WINDOW_NO_COLLAPSE)
        
        # Header
        imgui.text(f"Video Browser - {len(self.filtered_videos)} files")
        
        # Search box
        imgui.push_item_width(self._scaled(400))
        changed, new_text = imgui.input_text("##search", self.search_buffer, 256)
        imgui.pop_item_width()
        if changed:
            self.search_buffer = new_text
            self.search_text   = new_text
            self._apply_filters()
        
        imgui.same_line()
        imgui.text("Search")
        
        imgui.same_line(spacing=self._scaled(50))
        imgui.text("Sort by:")
        
        imgui.same_line()
        
        # Sort dropdown - calculate width based on content
        sort_options = ["name", "rating", "size"]
        max_option_width = max(imgui.calc_text_size(opt)[0] for opt in sort_options)
        combo_width = max_option_width + self._scaled(40)  # Add padding for arrow
        
        imgui.push_item_width(combo_width)
        if imgui.begin_combo("##sort", self.sort_by):
            for option in sort_options:
                is_selected, _ = imgui.selectable(option, self.sort_by == option)
                if is_selected:
                    self.sort_by = option
                    self._apply_filters()
            imgui.end_combo()
        imgui.pop_item_width()
        
        imgui.separator()
        
        # Filters section (non-scrolling)
        filter_height = self._render_filters()
        
        imgui.separator()
        
        # Scrollable video grid
        self._render_video_grid()
        
        imgui.end()
    
    def _render_filters(self):
        """Render filter controls. Returns height used."""
        start_cursor_y = imgui.get_cursor_pos()[1]
        
        # Special filters
        #imgui.text("Filters:")
        #imgui.same_line()
        
        # Rating filter
        imgui.text("Min Rating:")
        imgui.same_line()
        
        # Draw interactive star rating
        draw_list = imgui.get_window_draw_list()
        star_size = self._scaled(12)
        star_spacing = star_size * 2.2
        
        for star in range(1, 6):
            cursor_screen_pos = imgui.get_cursor_screen_pos()
            
            # Invisible button for click detection
            is_active = self.min_rating == star and not self.filter_no_rating
            clicked = imgui.invisible_button(f"##star{star}", star_spacing, star_size * 1.5)
            
            if clicked:
                if self.min_rating == star:
                    self.min_rating = 0
                else:
                    self.min_rating = star
                    self.filter_no_rating = False
                self._apply_filters()
            
            # Draw star (filled if <= current min_rating, empty otherwise)
            star_x = cursor_screen_pos[0] + star_size
            star_y = cursor_screen_pos[1] + star_size * 0.75
            
            current_min = self.min_rating if not self.filter_no_rating else 0
            filled = (star <= current_min)
            
            if is_active:
                # Highlight the selected threshold
                color = (1.0, 0.9, 0.3, 1.0)
            elif filled:
                color = (0.9, 0.7, 0.2, 1.0)
            else:
                color = (0.4, 0.4, 0.4, 1.0)
            
            self._draw_star(draw_list, star_x, star_y, star_size, filled, color)
            
            imgui.same_line(spacing=self._scaled(2))
        
        if False:   # Clicking on currently selected star handles this.
            imgui.same_line()
            if imgui.button("Clear##rating"):
                self.min_rating = 0
                self._apply_filters()
        
        clicked, self.filter_no_tags = imgui.checkbox("No Tags", self.filter_no_tags)
        if clicked:
            if self.filter_no_tags:
                self.selected_tags.clear()
            self._apply_filters()
        
        imgui.same_line()
        
        clicked, self.filter_no_rating = imgui.checkbox("No Rating", self.filter_no_rating)
        if clicked:
            if self.filter_no_rating:
                self.min_rating = 0
            self._apply_filters()
        
        # Tag filters with wrapping
        if self.all_tags:
            #imgui.text("Tags:")
            imgui.text("")
            
            # Calculate available width for tags
            available_width = imgui.get_content_region_available()[0]
            
            # Render tags with wrapping
            cursor_x_start = imgui.get_cursor_pos()[0]
            current_line_width = 0
            first_tag = True
            
            for tag, count in self.all_tags[:50]:  # Limit displayed tags
                is_selected = tag in self.selected_tags
                
                # Calculate width needed for this tag checkbox
                tag_text = f"{tag} ({count})"
                tag_width = imgui.calc_text_size(tag_text)[0] + self._scaled(30)
                
                # Check if we need to wrap
                if current_line_width > 0 and current_line_width + tag_width > available_width:
                    current_line_width = 0
                    first_tag = True
                
                if not first_tag:
                    imgui.same_line()
                
                clicked, _ = imgui.checkbox(tag_text, is_selected)
                if clicked:
                    if is_selected:
                        self.selected_tags.discard(tag)
                    else:
                        self.selected_tags.add(tag)
                        self.filter_no_tags = False
                    self._apply_filters()
                
                current_line_width += tag_width
                first_tag = False

        end_cursor_y = imgui.get_cursor_pos()[1]
        return end_cursor_y - start_cursor_y
    
    def _render_video_grid(self):
        """Render scrollable video grid with virtual scrolling."""
        # Calculate grid layout
        available_width, list_height = imgui.get_content_region_available()
        
        scaled_min_width = self._scaled(self.min_column_width)
        num_columns = max(1, int(available_width / scaled_min_width))
        column_width = available_width / num_columns
        
        # Use child window for scrolling
        imgui.begin_child("video_grid", 0, list_height, border=True)
        
        # Calculate item dimensions (portrait orientation)
        item_width = column_width - self._scaled(10)
        
        # Estimate row height for virtual scrolling
        estimated_row_height = item_width * 1.5  # Rough estimate
        
        # Virtual scrolling - only render visible rows
        scroll_y = imgui.get_scroll_y()
        visible_start_row = max(0, int(scroll_y / estimated_row_height) - 1)
        visible_end_row = min(
            (len(self.filtered_videos) + num_columns - 1) // num_columns,
            int((scroll_y + list_height) / estimated_row_height) + 2
        )
        
        # Add spacer for rows before visible range
        if visible_start_row > 0:
            imgui.dummy(0, visible_start_row * estimated_row_height)
        
        # Render visible rows
        row_start_y = imgui.get_cursor_pos()[1]
        max_row_height = 0
        current_row = visible_start_row
        
        start_idx = visible_start_row * num_columns
        end_idx = min(len(self.filtered_videos), visible_end_row * num_columns)
        
        for i in range(start_idx, end_idx):
            video = self.filtered_videos[i]
            col = i % num_columns
            row = i // num_columns
            
            # Start new row
            if row != current_row:
                imgui.set_cursor_pos((imgui.get_cursor_pos()[0], row_start_y + max_row_height + self._scaled(5)))
                row_start_y = imgui.get_cursor_pos()[1]
                max_row_height = 0
                current_row = row
            elif col > 0:
                imgui.same_line(spacing=self._scaled(5))
            
            # Render card and track height
            card_height = self._render_video_card(video, item_width)
            max_row_height = max(max_row_height, card_height)
        
        # Add spacer for rows after visible range
        total_rows = (len(self.filtered_videos) + num_columns - 1) // num_columns
        if visible_end_row < total_rows:
            remaining_rows = total_rows - visible_end_row
            imgui.dummy(0, remaining_rows * estimated_row_height)
        
        # Handle scroll-to-hash (for back navigation)
        if self.scroll_to_hash:
            for i, video in enumerate(self.filtered_videos):
                if video['hash'] == self.scroll_to_hash:
                    row = i // num_columns
                    target_scroll = row * estimated_row_height
                    imgui.set_scroll_y(target_scroll)
                    break
            self.scroll_to_hash = None
        
        imgui.end_child()

    def _render_video_card(self, video, width):
        """Render a single video card in portrait format. Returns actual height used."""
        # Card starts here
        start_pos = imgui.get_cursor_screen_pos()
        start_cursor_y = imgui.get_cursor_pos()[1]
        draw_list = imgui.get_window_draw_list()
        
        # Thumbnail size (square, full width)
        thumb_size = width
        info_padding = self._scaled(8)
        
        # Begin vertical layout
        imgui.begin_group()
        
        # Draw thumbnail placeholder or actual thumbnail
        cursor_screen_pos = imgui.get_cursor_screen_pos()
        if (texture_id := self._get_thumbnail_texture(video)):
            # Draw actual thumbnail
            draw_list.add_image(texture_id,
                               (cursor_screen_pos[0], cursor_screen_pos[1]),
                               (cursor_screen_pos[0] + thumb_size, cursor_screen_pos[1] + thumb_size))
        else:
            # Draw placeholder
            placeholder_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.25, 1.0)
            draw_list.add_rect_filled(cursor_screen_pos[0], cursor_screen_pos[1],
                                      cursor_screen_pos[0] + thumb_size, 
                                      cursor_screen_pos[1] + thumb_size,
                                      placeholder_color)
        
        # Reserve space for thumbnail
        imgui.dummy(thumb_size, thumb_size)
        
        # Info section below thumbnail
        imgui.spacing()
        
        # Add padding on sides for info
        imgui.indent(info_padding)
        info_width = width - info_padding * 2
        
        # Line 1: File name (wrapped)
        imgui.push_text_wrap_pos(imgui.get_cursor_pos()[0] + info_width)
        imgui.text(video['stem'])
        imgui.pop_text_wrap_pos()
        
        # Line 2: Size and rating on same line
        imgui_text_colored((0.6, 0.6, 0.6, 1.0), self._format_size(video['size']))
        imgui.same_line(spacing=self._scaled(10))
        self._draw_rating_stars(video['rating'], size=self._scaled(8))
        
        # Remaining lines: Tags (wrapped)
        if video['path_tags'] or video['tags']:
            imgui.spacing()
            
            # Manual wrapping for tags
            available_tag_width = info_width
            current_line_width = 0
            first_tag = True
            
            all_tags = [(f"[{tag}]", (0.4, 0.5, 0.6, 1.0)) for tag in video['path_tags']] + \
                       [(f"#{tag}", (0.3, 0.6, 0.8, 1.0)) for tag in video['tags']]
            
            for tag_text, tag_color in all_tags:
                tag_width = imgui.calc_text_size(tag_text)[0] + self._scaled(4)
                
                # Check if we need to wrap to next line
                if current_line_width > 0 and current_line_width + tag_width > available_tag_width:
                    # Force wrap by NOT using same_line
                    current_line_width = 0
                    first_tag = True
                
                if not first_tag:
                    # Continue on same line
                    imgui.same_line(spacing=self._scaled(4))
                
                imgui_text_colored(tag_color, tag_text)
                current_line_width += tag_width
                first_tag = False

        imgui.unindent(info_padding)
        imgui.spacing()
        
        imgui.end_group()
        
        # Calculate total height used
        end_cursor_y = imgui.get_cursor_pos()[1]
        card_height = end_cursor_y - start_cursor_y
        
        # Make the whole card clickable (overlay invisible button)
        end_pos = imgui.get_cursor_screen_pos()
        imgui.set_cursor_screen_pos(start_pos)
        clicked = imgui.invisible_button(f"##card_{video['hash']}", width, card_height)
        if clicked:
            self.selected_video = video
            VideoBrowserPlugin.last_selected_video = video
        
        # Restore cursor to end of card
        imgui.set_cursor_pos((imgui.get_cursor_pos()[0], end_cursor_y))
        
        return card_height

    def go_back(self):
        if self.selected_video:
            self.scroll_to_hash = self.selected_video['hash']
            self.selected_video = None
            return True
        return False

    def _render_detail_view(self):
        """Render detail view for selected video."""
        play  = False   # Return value of this function -- did user click play?
        video = self.selected_video
        
        # Full window
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(*self.size)
        
        imgui.begin("Video Detail",
                   flags=imgui.WINDOW_NO_TITLE_BAR |
                         imgui.WINDOW_NO_RESIZE |
                         imgui.WINDOW_NO_MOVE |
                         imgui.WINDOW_NO_COLLAPSE)
        
        # Back button
        if imgui.button("<- Back"):
            self.go_back()
        
        imgui.separator()
        
        # Thumbnail preview (large square on left side)
        draw_list = imgui.get_window_draw_list()
        thumb_size = self._scaled(300)  # Large preview
        
        cursor_screen_pos = imgui.get_cursor_screen_pos()
        
        texture_id = self._get_thumbnail_texture(video)
        if texture_id:
            # Draw actual thumbnail
            draw_list.add_image(texture_id,
                               (cursor_screen_pos[0], cursor_screen_pos[1]),
                               (cursor_screen_pos[0] + thumb_size, cursor_screen_pos[1] + thumb_size))
        else:
            # Draw placeholder
            placeholder_color = imgui.get_color_u32_rgba(0.2, 0.2, 0.25, 1.0)
            draw_list.add_rect_filled(cursor_screen_pos[0], cursor_screen_pos[1],
                                      cursor_screen_pos[0] + thumb_size, 
                                      cursor_screen_pos[1] + thumb_size,
                                      placeholder_color)
        
        if imgui.invisible_button(f"thumbnail_{id(video)}", thumb_size, thumb_size):
            # User clicked on the thumbnail, which is the Play button
            play = True
        
        imgui.same_line(spacing=self._scaled(20))
        
        # Info column (to the right of thumbnail)
        imgui.begin_group()
        
        # Video title
        imgui.text(video['stem'])
        imgui_text_colored((0.6, 0.6, 0.6, 1.0), self._format_size(video['size']))
        
        imgui.separator()
        
        # Rating editor
        imgui.text("Rating:")
        imgui.same_line()
        
        current_rating = video['rating']
        draw_list = imgui.get_window_draw_list()
        star_size = self._scaled(12)
        star_spacing = star_size * 2.2
        
        for star in range(1, 6):
            cursor_screen_pos = imgui.get_cursor_screen_pos()
            
            # Invisible button for click detection
            clicked = imgui.invisible_button(f"##star{star}detail", star_spacing, star_size * 1.5)
            
            if clicked:
                self._set_rating(video['hash'], star)
                video['rating'] = star
            
            # Draw star (filled if <= current rating)
            star_x = cursor_screen_pos[0] + star_size
            star_y = cursor_screen_pos[1] + star_size * 0.75
            
            filled = (star <= current_rating)
            
            if star == current_rating:
                # Highlight the current rating
                color = (1.0, 0.9, 0.3, 1.0)
            elif filled:
                color = (0.9, 0.7, 0.2, 1.0)
            else:
                color = (0.4, 0.4, 0.4, 1.0)
            
            self._draw_star(draw_list, star_x, star_y, star_size, filled, color)
            
            imgui.same_line(spacing=self._scaled(2))
        
        imgui.new_line()  # End the star row
        
        imgui.spacing()
        imgui.separator()
        
        # Tags display and editing
        imgui.text("Tags:")
        
        # Current tags
        tags_to_remove = []
        for tag in video['tags']:
            imgui.same_line()
            if imgui.button(f"{tag} x"):
                tags_to_remove.append(tag)
        
        # Remove tags
        for tag in tags_to_remove:
            self._remove_tag(video['hash'], tag)
            video['tags'].remove(tag)
        
        # Add tag input
        imgui.spacing()
        imgui.text("Add tag:")
        imgui.same_line()
        
        imgui.push_item_width(self._scaled(200))
        changed, self.new_tag_buffer = imgui.input_text("##newtag", 
                                                         self.new_tag_buffer, 
                                                         256)
        imgui.pop_item_width()
        
        imgui.same_line()
        if imgui.button("Add"):
            if self.new_tag_buffer.strip():
                self._add_tag(video['hash'], self.new_tag_buffer.strip())
                video['tags'].append(self.new_tag_buffer.strip())
                self.new_tag_buffer = ''
        
        # Suggested tags (wrapped)
        imgui.spacing()
        imgui.text("Suggested:")
        
        available_width = imgui.get_content_region_available()[0]
        current_tags = set(video['tags'])
        
        current_line_width = 0
        first_tag = True
        
        for tag, count in self.all_tags:
            if tag not in current_tags:
                tag_width = imgui.calc_text_size(f"{tag}")[0] + self._scaled(20)
                
                if current_line_width > 0 and current_line_width + tag_width > available_width:
                    current_line_width = 0
                    first_tag = True
                
                if not first_tag:
                    imgui.same_line()
                
                if imgui.button(f"{tag}##suggest"):
                    self._add_tag(video['hash'], tag)
                    video['tags'].append(tag)
                
                current_line_width += tag_width
                first_tag = False

        if False:
            imgui.separator()
        
            # Actions
            if imgui.button("Play"):
                self._trigger_action('play', video)
            
            imgui.same_line()
            if imgui.button("Play 3D (SBS)"):
                self._trigger_action('play_3d_sbs', video)
            
            imgui.same_line()
            if imgui.button("Play 3D (OU)"):
                self._trigger_action('play_3d_ou', video)
        
            imgui.same_line(spacing=self._scaled(50))
            if imgui.button("Delete"):
                # Simple confirmation - just require double-click
                self._delete_video(video['hash'])
                self.selected_video = None
        
        imgui.end_group()
        imgui.separator()
        imgui.end()

        return play
    
    def _set_rating(self, file_hash, rating):
        """Update rating in database."""
        if self.readonly:
            self.log(f"READONLY: Would set {file_hash} rating to {rating}")
        else:
            if self.db_conn:
                self.db_conn.execute(
                    'UPDATE file_info SET rating = ? WHERE hash = ?',
                    (rating, file_hash)
                )
                self.db_conn.commit()
    
    def _add_tag(self, file_hash, tag):
        """Add tag to video in database."""
        if not self.db_conn:
            return
        
        row = self.db_conn.execute(
            'SELECT tags FROM file_info WHERE hash = ?',
            (file_hash,)
        ).fetchone()
        
        if row:
            tags = row['tags'].split('|') if row['tags'] else []
            if tag not in tags:
                if self.readonly:
                    self.log(f"READONLY: Would add tag {tag!r} to {file_hash}")
                else:
                    tags.append(tag)
                    self.db_conn.execute(
                        'UPDATE file_info SET tags = ? WHERE hash = ?',
                        ('|'.join(tags), file_hash)
                    )
                    self.db_conn.commit()
                    self._refresh_file_list()
    
    def _remove_tag(self, file_hash, tag):
        """Remove tag from video in database."""
        if not self.db_conn:
            return
        
        row = self.db_conn.execute(
            'SELECT tags FROM file_info WHERE hash = ?',
            (file_hash,)
        ).fetchone()
        
        if row:
            tags = row['tags'].split('|') if row['tags'] else []
            if tag in tags:
                if self.readonly:
                    self.log(f"READONLY: Would remove tag {tag!r} from {file_hash}")
                else:
                    tags.remove(tag)
                    self.db_conn.execute(
                        'UPDATE file_info SET tags = ? WHERE hash = ?',
                        ('|'.join(tags), file_hash)
                    )
                    self.db_conn.commit()
                    self._refresh_file_list()
    
    if False:
        def _delete_video(self, file_hash):
            """Delete video (move to trash)."""
            # This would integrate with your Organizer.trash_file()
            # For now, just remove from database
            if self.readonly:
                self.log(f"READONLY: Would delete video {file_hash} from db")
            else:
                self.trash_file(NEED_FULL_PATH_HERE)    # FIXME
                if self.db_conn:
                    self.db_conn.execute(
                        'DELETE FROM file_info WHERE hash = ?',
                        (file_hash,)
                    )
                    self.db_conn.commit()
                    self._refresh_file_list()
    
    # ============= Thumbnails =================

    def _get_thumbnail_texture(self, video):
        """Get OpenGL texture for thumbnail, loading if needed. Returns texture ID or None."""
        file_hash = video['hash']
        
        # Check cache
        if file_hash in self.thumbnail_cache:
            return self.thumbnail_cache[file_hash]
        
        # Check if known missing
        if file_hash in self.thumbnails_missing:
            return None
        
        # Check if already loading
        if file_hash in self.thumbnails_loading:
            return None
        
        # Find thumbnail file (paths are relative to video_root where the db is)
        video_path = self.video_root / video['path']
        jpg_path = video_path.with_suffix('.jpg')
        
        if not jpg_path.exists():
            # Mark as missing so we don't check again
            self.thumbnails_missing.add(file_hash)
            print(f"Missing: {jpg_path}")
            return None
        
        # Queue for loading
        import threading
        with self.thumbnail_load_lock:
            self.thumbnail_load_queue.append((file_hash, str(jpg_path)))
            self.thumbnails_loading.add(file_hash)
            
            # Start loader thread if not running
            if self.thumbnail_loader_thread is None:
                self.thumbnail_loader_thread = threading.Thread(
                    target=self._thumbnail_loader_worker, daemon=True
                )
                self.thumbnail_loader_thread.start()
        
        return None

    def _thumbnail_loader_worker(self):
        """Background thread that loads thumbnails."""
        from PIL import Image
        
        while True:
            # Get next item to load
            with self.thumbnail_load_lock:
                if not self.thumbnail_load_queue:
                    self.thumbnail_loader_thread = None
                    return
                file_hash, jpg_path = self.thumbnail_load_queue.pop(0)
            
            try:
                # Load image
                img = Image.open(jpg_path)
                img = img.convert('RGB')
                # Optionally downsize the thumbnail (not needed if originals are good thumbnail size)
                if False:
                    img.thumbnail((256, 256), Image.Resampling.LANCZOS)
                
                width, height = img.size
                
                # Convert to bytes directly from PIL (correct format for OpenGL)
                img_bytes = img.tobytes()
                
                # Store for main thread to upload to GPU
                with self.thumbnail_load_lock:
                    if not hasattr(self, '_loaded_thumbnails'):
                        self._loaded_thumbnails = []
                    self._loaded_thumbnails.append((file_hash, img_bytes, width, height))
                
            except Exception as e:
                print(f"Failed to load thumbnail {jpg_path}: {e}")
                with self.thumbnail_load_lock:
                    self.thumbnails_loading.discard(file_hash)

    # ============= Database =================

    def _init_database(self, index=True, create=False):
        """Opens the database, and optionally indexes any new files.
        """
        self.init_db(create)

        if index:
            self.index_directory()

    def init_db(self, create=False):
        """Initialize database schema with migrations.
        """
        if self.db_conn is not None:
            return

        if not self.db_path.exists():
            if self.readonly:
                raise Exception(f"{self.db_path} doesn't exist but we can't create it in read-only mode.")
            if not create:
                raise Exception(f"{self.db_path} doesn't exist.  Set Create flag (-C) to initialize.")

        self.db_conn             = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row

        if self.readonly:
            self.log(f"READONLY: Would migrate the database as needed...")
            return

        conn = self.db_conn
        
        # Create version table if it doesn't exist
        conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            )
        ''')
            
        # Get current schema version
        row = conn.execute('SELECT version FROM schema_version').fetchone()
        current_version = row[0] if row else 0

        # Define migrations
        migrations = [
            # Version 1: Initial schema
            lambda c: (
                c.execute('''
                    CREATE TABLE IF NOT EXISTS file_info (
                        hash TEXT PRIMARY KEY,
                        file_path TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        mod_time REAL NOT NULL,
                        tags TEXT
                    )
                '''),
                c.execute('''
                    CREATE INDEX IF NOT EXISTS idx_path_size_time 
                    ON file_info (file_path, file_size, mod_time)
                ''')
            ),
            # Version 2: Add alt_paths field for tracking moved/duplicate files
            lambda c: c.execute('ALTER TABLE file_info ADD COLUMN alt_paths TEXT'),
            # Version 3: Add rating field (0-5, 0 means unrated)
            lambda c: c.execute('ALTER TABLE file_info ADD COLUMN rating INTEGER DEFAULT 0'),
        ]

        # Run any migrations needed
        for version, migration in enumerate(migrations, 1):
            if version > current_version:
                migration(conn)
                conn.execute('DELETE FROM schema_version')
                conn.execute('INSERT INTO schema_version (version) VALUES (?)', (version,))
                conn.commit()
                print(f"Migrated database to schema version {version}")

        conn.commit()

    #
    # Although this returns a dict, at the moment it's really just used to
    #  scan the directory for changes.
    #
    def index_directory(self, directory=None):

        if directory is None:
            directory = self.video_root

        if self.readonly:
            self.log(f"READONLY: Would index {directory}")
            return

        videos = {} # Maps hash to info dict

        if directory.exists():
            for item in list(directory.iterdir()):
                if item.is_file() and item.suffix.lower() in self.ALLOWED_EXTENSIONS:
                    # Skip empty/placeholder files
                    if item.stat().st_size == 0:
                        continue
                    rel_path = item.relative_to(self.video_root)
                    info = self.get_file_info(rel_path=str(rel_path))
                    if info:
                        videos[info['hash']] = info
                elif item.is_dir() and not item.is_symlink() and item != self.trash_dir:
                    videos.update(self.index_directory(item))
        return videos

    def add_path(self, alt_paths, path):
        return list(set(alt_paths)|{str(path).replace('|','')})

    def tags_to_string(self, tags):
        """Convert list of tags to database string format."""
        return '|'.join(tags) if tags else ''
    
    def tags_from_string(self, tags_str):
        """Convert database string format to list of tags."""
        return tags_str.split('|') if tags_str else []
    
    def hash_file(self, filepath, chunk_size=65536):
        """Compute xxh128 hash of file."""
        self.log(f"Hashing: {filepath}")
        h = xxhash.xxh128()
        try:
            with open(filepath, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    h.update(chunk)
            result = h.hexdigest()
            self.log(f"Hashed: {filepath} -> {result}")
            return result
        except Exception as e:
            self.log(f"Hash failed: {filepath} ({e})")
            return None
    
    def trash_file(self, full_path):
        if self.readonly:
            self.log("READONLY: Would TRASH {full_path}")
            return
        self.trash_dir.mkdir(parents=True, exist_ok=True)
        trash_path = self.trash_dir / full_path.name
        n = 0
        while trash_path.exists() or trash_path.is_symlink():
            n = n + 1
            trash_path = self.trash_dir / f"{full_path.name}.{n}"
        self.log(f"Moving {full_path} to {trash_path}")
        full_path.rename(trash_path)

    def get_file_info(self, rel_path=None, file_hash=None):
        """
        Get file info by (db root) relative path or hash.
        Exactly one of rel_path or hash must be provided.
        Returns dict with keys: name, stem, size, tags, path_tags, rel_path, full_path, hash
        Returns None if the file can't be found.
        New or changed files (via rel_path param) will be inducted into the db, possibly removing duplicates.
        NOTE that if rel_path is a duplicate, in the event that it is the one that is removed, None will
            be returned since the file no longer exists.  It is assumed you will encounter the retained
            version later if you haven't already.  This means the returned file info will always point
            to the same rel_path provided.
        NOTE furthermore that some files may be identified as duplicates and removed after you have
            already gotten their file info, in which case those old records become invalid.  You
            should index your info by hash and always replace with the most recent file_info.
        """
        assert not self.readonly    # Nobody should call this method in readonly mode.
        assert (rel_path is None) != (file_hash is None), "Exactly one of rel_path or file_hash must be provided"
        
        conn = self.db_conn
        
        #
        # If a file_hash is provided, we'll just translate that to a rel_path and
        #  then proceed more or less down the rel_path chain, since we still
        #  need to check that the actual file matches expectations.  This results
        #  in two db queries where we could just do one, but oh well -- much
        #  simpler code, and the majority of calls, due to directory traversal,
        #  provide rel_path directly:
        #
        if file_hash is not None:
            row = conn.execute(
                'SELECT file_path FROM file_info WHERE hash = ?',
                (file_hash,)
            ).fetchone()
            
            if not row:
                return None
            
            rel_path = Path(row[0])
        else:
            rel_path = Path(rel_path)
 
        full_path = self.video_root / rel_path

        #
        # Verify file exists and is an ordinary file and not a sym link:
        #
        if full_path.is_symlink() or not full_path.exists() or not full_path.is_file():
            return None

        stat      = full_path.stat()
        file_size = stat.st_size
        mod_time  = stat.st_mtime

        # Check if we have a cached entry with matching path/size/time
        row = conn.execute(
            '''SELECT hash, tags, alt_paths, rating FROM file_info 
               WHERE file_path = ? AND file_size = ? AND mod_time = ?''',
            (str(rel_path), file_size, mod_time)
        ).fetchone()
        
        if row:
            #
            # Cache hit.  Extract file hash and tags and we're done.
            #
            if file_hash is not None and file_hash != row[0]:   # Obscure but possible case that's a false cache hit for the actual request.
                return None

            file_hash = row[0]
            tags      = self.tags_from_string(row[1])
            alt_paths = row[2].split('|') if row[2] else []
        else:
            print(f"Cache miss on {rel_path} / {file_hash}:")

            #
            # We don't immediately recognize this file, so we need to (re-)hash it and
            #  see if it's new or moved.
            #
            provided_file_hash = file_hash

            if (file_hash := self.hash_file(full_path)) is None:
                return None
            
            # If we're searching by hash but found a different file, it's a miss:
            if provided_file_hash is not None and file_hash != provided_file_hash:
                print("  There's a new file where old one with this hash was")
                return None

            # Check if this hash exists in the database
            prior = conn.execute(
                'SELECT file_path, file_size, mod_time, alt_paths, tags FROM file_info WHERE hash = ?',
                (file_hash,)
            ).fetchone()
            
            #
            # Handle the simple cases first.
            #
            # We've never seen a file with this hash:
            #
            if not prior:
                print("  This is a new file")
                # New file, insert it
                conn.execute(
                    '''INSERT INTO file_info (hash, file_path, file_size, mod_time, tags, alt_paths, rating)
                       VALUES (?, ?, ?, ?, ?, ?, ?)''',
                    (file_hash, str(rel_path), file_size, mod_time, '', '', 0)
                )
                conn.commit()
                tags      = []
                alt_paths = []

            #
            # We have seen this file (hash), so extract what we know:
            #
            else:
                prior_rel_path_str, prior_size, prior_mod_time, alt_paths_str, tags_str = prior

                prior_rel_path  = Path(prior_rel_path_str)
                prior_full_path = self.video_root / prior_rel_path
                tags            = self.tags_from_string(tags_str)
                alt_paths       = alt_paths_str.split('|') if alt_paths_str else []

                assert file_size == prior_size   # Hash match should assure this

                #
                # Next simple case:
                #
                # We have seen a file with this hash, and it's still here, just with a new mod_time:
                #
                if rel_path == prior_rel_path:
                    print("   Only the mod time changed")
                    # Then just update the mod time in the database but otherwise treat it as a cache hit:
                    conn.execute(
                        '''UPDATE file_info 
                           SET mod_time = ?
                           WHERE hash = ?''',
                        (mod_time, file_hash)
                    )
                    conn.commit()

                #
                # If the path changed, could be it's moved, or could be we've discovered a duplicate.
                # If it's a duplicate, we have to decide which to keep.
                #
                else:
                    print(f"   We know this file as {prior_rel_path}")
                    
                    #
                    # Simple case -- it's moved.  Update the cache, and fall through.
                    #
                    if not prior_full_path.exists() or self.hash_file(prior_full_path) != file_hash:
                        print("   which is no longer there, so we'll note it as moved.")
                        alt_paths = self.add_path(alt_paths, prior_rel_path)
                        conn.execute(
                            '''UPDATE file_info 
                               SET file_path = ?, mod_time = ?, alt_paths = ?
                               WHERE hash = ?''',
                            (str(rel_path), mod_time, '|'.join(alt_paths), file_hash)
                        )
                        conn.commit()

                    #
                    # Seems we have two versions of the file at two different locations.
                    #
                    else:
                        if mod_time > prior_mod_time:
                            # Keep prior path (older), trash the requested one:
                            print(f"   Which is older, so we'll delete {rel_path} and retain {prior_rel_path}")
                            self.trash_file(full_path)
                            alt_paths = self.add_path(alt_paths, rel_path)
                            conn.execute(
                                'UPDATE file_info SET alt_paths = ? WHERE hash = ?',
                                ('|'.join(alt_paths), file_hash)
                            )
                            conn.commit()
                            return None     # The file requested no longer exists!
                        else:
                            # Keep this path (which is older), trash prior, and fall through
                            print(f"   Which is newer, so we'll delete {prior_rel_path} and treat it as moved to {rel_path}.")
                            self.trash_file(prior_full_path)
                            alt_paths = self.add_path(alt_paths, prior_rel_path)
                            conn.execute(
                                '''UPDATE file_info 
                                   SET file_path = ?, mod_time = ?, alt_paths = ?
                                   WHERE hash = ?''',
                                (str(rel_path), mod_time, '|'.join(alt_paths), file_hash)
                            )
                            conn.commit()
        
        # Construct the return dict
        return {
            'name'     : full_path.name,
            'stem'     : full_path.stem,
            'ext'      : full_path.suffix[1:] if full_path.suffix else '',
            'size'     : file_size,
            'tags'     : tags,
            'path_tags': list(rel_path.parent.parts),
            'rel_path' : str(rel_path),
            'full_path': full_path,
            'hash'     : file_hash,
            'alt_paths': alt_paths,
            'rating'   : row[3] if row else 0,
        }
    
    # ============= Misc =================

    def log(self, msg):
        print(msg)

    def close(self):
        """Cleanup resources."""
        if self.db_conn:
            self.db_conn.close()
        
        if self.fbo:
            glDeleteFramebuffers(1, [self.fbo])
        if self.depth_rbo:
            glDeleteRenderbuffers(1, [self.depth_rbo])
        
        if self.renderer:
            self.renderer.shutdown()
        
        imgui.destroy_context(self.context)

def imgui_text_colored(rgba, text):
    imgui.text_colored(text, *rgba)

