import pyaudio
import numpy as np
from time import perf_counter as time
import sys
import os
from contextlib import contextmanager

@contextmanager
def suppress_c_stderr():
    """Suppress C-level stderr output (PortAudio messages) while preserving Python stderr.
    
    This redirects file descriptor 2 temporarily, which catches messages from C libraries
    like PortAudio, but Python's sys.stderr remains functional for exceptions/tracebacks.
    """
    # Flush Python's stderr first to avoid losing buffered content
    sys.stderr.flush()
    
    # Save the original stderr file descriptor
    stderr_fd = 2
    saved_stderr = os.dup(stderr_fd)
    
    try:
        # Redirect stderr to devnull at the OS level
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
        os.close(devnull_fd)
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stderr)

class AudioPlay(object):
    def __init__(self, audio_data, fps, sample_rate=48000):
        """
        Args:
            audio_data: numpy array, shape (samples,) or (samples, channels)
            fps: video frame rate
            sample_rate: audio sample rate
        """
        self.audio_data   = audio_data
        self.fps          = fps
        self.sample_rate  = sample_rate
        
        # Infer channels from audio data shape
        if audio_data.ndim == 1:
            self.channels   = 1
            self.audio_data = audio_data.reshape(-1, 1)
        else:
            self.channels = audio_data.shape[-1]
        
        # Playback state
        self.audio_position = 0  # Samples actually sent to audio hardware (can go beyond bounds)
        self.pause_position = 0  # Position where we're paused (or will start)
        self.paused         = True
        self.stop_flag      = False
        self.speed          = 1.0
        
        # For dead reckoning between callbacks
        self.last_callback_time     = None
        self.last_callback_position = 0
        
        # Audio setup
        with suppress_c_stderr():
            self.p      = pyaudio.PyAudio()
        self.stream = None
    
    def set_speed(self, value):
        """Set playback speed (negative for reverse, 0 pauses, positive for forward)."""
        self.speed = value
    
    def get_audio_time(self):
        """Get time based on actual audio samples played (source position / sample rate).
        Uses dead reckoning when playing to estimate position between callbacks.
        Note: Position can go beyond array bounds - this is intentional for proper timing."""
        if self.paused or self.speed == 0:
            position = self.pause_position
        else:
            # Dead reckon: estimate current position based on time since last callback
            if self.last_callback_time is not None:
                elapsed_wall_time = time() - self.last_callback_time
                # How many source samples have we advanced since last callback?
                samples_advanced = elapsed_wall_time * self.sample_rate * self.speed
                estimated_position = self.last_callback_position + samples_advanced
                # NO CLAMPING - allow position to go beyond bounds
                position = estimated_position
            else:
                position = self.audio_position
        
        return position / self.sample_rate
    
    def get_current_frame(self):
        """Get which video frame should be displayed now."""
        return int(self.get_audio_time() * self.fps)
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - feeds sequential audio samples."""
        current_time = time()
        
        if self.paused or self.speed == 0:
            # Return silence but don't advance position
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(), 
                    pyaudio.paContinue)
        
        # Track timing for dead reckoning
        self.last_callback_time     = current_time
        self.last_callback_position = self.audio_position
        
        # Determine direction and absolute speed
        direction = 1 if self.speed > 0 else -1
        abs_speed = abs(self.speed)
        
        # Calculate how many source samples we need for target frame count at current speed
        source_samples_needed = int(frame_count * abs_speed)
        
        if source_samples_needed == 0:
            # Update position even for zero speed (allows time to advance)
            self.audio_position += direction * source_samples_needed
            return (np.zeros((frame_count, self.channels), dtype=np.float32).tobytes(),
                    pyaudio.paContinue)
        
        # Get source samples based on direction - handle out-of-bounds gracefully
        if direction > 0:
            # Forward playback
            start_pos = self.audio_position
            end_pos   = self.audio_position + source_samples_needed
            
            # Clamp to valid array range for reading
            read_start = max(0, min(start_pos, len(self.audio_data)))
            read_end   = max(0, min(end_pos, len(self.audio_data)))
            
            if read_end > read_start:
                source_chunk = self.audio_data[read_start:read_end]
            else:
                source_chunk = np.zeros((0, self.channels), dtype=np.float32)
                
        else:
            # Reverse playback
            start_pos = self.audio_position - source_samples_needed
            end_pos   = self.audio_position
            
            # Clamp to valid array range for reading
            read_start = max(0, min(start_pos, len(self.audio_data)))
            read_end   = max(0, min(end_pos, len(self.audio_data)))
            
            if read_end > read_start:
                source_chunk = self.audio_data[read_start:read_end]
                # Reverse the chunk for backward playback
                source_chunk = source_chunk[::-1]
            else:
                source_chunk = np.zeros((0, self.channels), dtype=np.float32)
        
        # Resample to output frame count
        if len(source_chunk) == 0:
            # All silence
            audio_chunk = np.zeros((frame_count, self.channels), dtype=np.float32)
        elif len(source_chunk) < source_samples_needed:
            # Partial data - pad with silence rather than stretching
            # Calculate how much valid data we should output
            valid_frames = int(len(source_chunk) / abs_speed)
            valid_frames = min(valid_frames, frame_count)
            
            if valid_frames > 0:
                source_indices = np.linspace(0, len(source_chunk) - 1, valid_frames)
                source_indices = np.clip(source_indices.astype(int), 0, len(source_chunk) - 1)
                valid_chunk = source_chunk[source_indices]
            else:
                valid_chunk = np.zeros((0, self.channels), dtype=np.float32)
            
            # Pad the rest with silence
            silence = np.zeros((frame_count - valid_frames, self.channels), dtype=np.float32)
            audio_chunk = np.vstack([valid_chunk, silence]) if valid_frames > 0 else silence
        else:
            # Normal case - full resampling
            source_indices = np.linspace(0, len(source_chunk) - 1, frame_count)
            source_indices = np.clip(source_indices.astype(int), 0, len(source_chunk) - 1)
            audio_chunk = source_chunk[source_indices]
        
        # Update position by the REQUESTED amount (not actual samples read)
        # This keeps timing consistent even when playing silence
        self.audio_position += direction * source_samples_needed
        
        # NO CLAMPING - position can go beyond array bounds
        
        return (audio_chunk.tobytes(), pyaudio.paContinue)
    
    def pause(self):
        """Pause playback (idempotent)."""
        if not self.paused:
            # Use dead reckoning to estimate current position
            if self.last_callback_time is not None:
                elapsed_wall_time = time() - self.last_callback_time
                samples_advanced = int(elapsed_wall_time * self.sample_rate * self.speed)
                estimated_position = self.last_callback_position + samples_advanced
                self.pause_position = estimated_position  # No clamping
            else:
                self.pause_position = self.audio_position
            
            self.paused = True
            
            # Stop the stream
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
    
    def resume(self):
        """Resume playback (idempotent)."""
        if self.paused:
            # Set position to where we paused
            self.audio_position = self.pause_position
            
            # Create fresh stream (clears any lingering buffers)
            with suppress_c_stderr():
                self.stream = self.p.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    stream_callback=self.audio_callback,
                    frames_per_buffer=1024
                )
                self.stream.start_stream()
            
            self.paused                 = False
            self.last_callback_time     = time()
            self.last_callback_position = self.audio_position
    
    def close(self):
        """Close and cleanup audio resources."""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        with suppress_c_stderr():
            self.p.terminate()
    
    def at_end(self):
        """Check if we've reached the end (forward) or beginning (reverse) of the audio."""
        if self.speed >= 0:
            return self.audio_position >= len(self.audio_data)
        else:
            return self.audio_position <= 0
    
    def seek(self, time_seconds):
        """Seek to specific time (works whether playing or paused).
        Can seek beyond array bounds - will play silence there."""
        was_playing = not self.paused
        
        # Pause first to get clean state
        if was_playing:
            self.pause()
        
        # Set new position (time is always absolute, not speed-adjusted)
        # Allow seeking beyond bounds
        new_position        = int(time_seconds * self.sample_rate)
        self.audio_position = new_position
        self.pause_position = new_position
        
        # Resume if was playing
        if was_playing:
            self.resume()


# Example usage
if __name__ == "__main__":
    # Create dummy audio and video
    duration    = 5.0
    sample_rate = 48000
    fps         = 30
    
    # Generate test audio (stereo sine wave)
    t           = np.linspace(0, duration, int(duration * sample_rate))
    audio_left  = np.sin(2 * np.pi * 440 * t) * 0.3  # A4 note
    audio_right = np.sin(2 * np.pi * 554 * t) * 0.3  # C#5 note
    audio       = np.column_stack([audio_left, audio_right]).astype(np.float32)
    
    # Create audio player
    player = AudioPlay(audio, fps=fps, sample_rate=sample_rate)
    
    print(f"Audio: {player.channels} channels, {len(audio)/sample_rate:.2f}s")
    print(f"Video: {fps} fps")
    
    # Start playback
    player.resume()
    
    try:
        last_frame = -1
        loop_count = 0
        
        while loop_count < 2:  # Play twice to test looping
            frame = player.get_current_frame()
            
            if frame != last_frame:
                print(f"Frame {frame:4d} @ {player.get_audio_time():.3f}s (speed={player.speed}x)", end='\r')
                last_frame = frame
            
            # Test going past the end
            if frame == 140 and loop_count == 0:
                print("\nPast end, playing silence...")
            
            # Test reversing after going past end
            if frame == 160 and loop_count == 0:
                print("\nReversing from beyond end...")
                player.set_speed(-1.0)
            
            # Check for end and loop
            if player.at_end():
                print(f"\nReached end, looping... (loop {loop_count + 1})")
                player.seek(0)
                player.set_speed(1.0)  # Reset speed
                loop_count += 1
                last_frame = -1
            
            import time as time_module
            time_module.sleep(0.001)
    
    finally:
        player.pause()
        player.close()
        print("\nDone!")

