
"""
Google Cardboard VR Panorama Reader

Loads Cardboard VR panoramas with embedded stereo images and audio into memory.
Images are loaded as GPU arrays (CuPy) and audio can be decompressed to numpy arrays.
"""
import Exif
import numpy                 as np
import cupy                  as cp
import subprocess            as sp
import xml.etree.ElementTree as ET

from PIL    import Image
from base64 import b64decode
from io     import BytesIO
from typing import Optional, Tuple


from VideoSource import VideoSource

class CardboardDecoder(VideoSource):

    """Loads and processes Google Cardboard VR panorama images with embedded audio."""
    
    # XML namespace mappings for Cardboard metadata
    _KEYS = {
        "{http://ns.google.com/photos/1.0/panorama/}CroppedAreaLeftPixels": ('crop_left', 'int'),
        "{http://ns.google.com/photos/1.0/panorama/}CroppedAreaTopPixels": ('crop_top', 'int'),
        "{http://ns.google.com/photos/1.0/panorama/}CroppedAreaImageWidthPixels": ('crop_width', 'int'),
        "{http://ns.google.com/photos/1.0/panorama/}CroppedAreaImageHeightPixels": ('crop_height', 'int'),
        "{http://ns.google.com/photos/1.0/panorama/}FullPanoWidthPixels": ('width', 'int'),
        "{http://ns.google.com/photos/1.0/panorama/}FullPanoHeightPixels": ('height', 'int'),
        "{http://ns.google.com/photos/1.0/image/}Data": ('image_data', 'base64'),
        "{http://ns.google.com/photos/1.0/audio/}Data": ('audio_data', 'base64'),
    }
    
    def __init__(self, video_path: str):
        VideoSource.__init__(self)

        self.filepath = video_path

        self._compressed_audio: Optional[bytes] = None
        self._vr360_image: Optional[cp.ndarray] = None
        
        # Load and process the panorama
        self._load_panorama()
    
        self.height, self.width, _ = self._vr360_image.shape
        self.framerate = 60
        self.duration  = self.audio_duration() or 1.0
        self.frame_count = int(self.duration * self.framerate)

    def get_frame(self, frame_number, io=None):
        if not 0 <= frame_number < self.frame_count:
            return None
        return self._vr360_image

    def raw_audio(self):
        """Decodes and returns the audio as a float32 numpy array.
        """
        return self.decompress_audio()

    #=======

    def _parse_xmp_node(self, node: ET.Element, props: dict):
        """Recursively parse XMP XML nodes."""
        for name, val in node.items():
            if name in self._KEYS:
                key, kind = self._KEYS[name]
                if kind == 'int':
                    val = int(val)
                elif kind == 'base64':
                    # Add padding if needed for base64
                    val = b64decode(val + '=' * ((4 - len(val)) % 4))
                props[key] = val
        
        for child in node:
            self._parse_xmp_node(child, props)
    
    def _load_panorama(self):
        """Load the panorama and extract left/right images and audio."""
        # Open the image
        with Image.open(self.filepath) as img:
            # Left image is the main image
            left_img = np.array(img.convert('RGB'))
        
        # Extract metadata using Exif library
        metadata = {}
        segs = Exif.get_segments(self.filepath)
        for seg in segs:
            if 'xmp' in seg:
                try:
                    root = ET.fromstring(seg['xmp'])
                    self._parse_xmp_node(root, metadata)
                except ET.ParseError:
                    pass
        
        # Extract right image from metadata
        right_data = metadata.get('image_data')
        if right_data:
            right_img = np.array(Image.open(BytesIO(right_data)).convert('RGB'))
        else:
            raise ValueError("No right eye image found in Cardboard panorama")
        
        # Extract compressed audio
        self._compressed_audio = metadata.get('audio_data')
        
        # Create VR360 TB (top-bottom) format with black padding
        self._vr360_image = self._create_vr360_tb(left_img, right_img)

    def _create_vr360_tb(self, left_img: np.ndarray, right_img: np.ndarray) -> cp.ndarray:
        """
        Create VR360 top-bottom format with proper equirectangular padding.
        
        Args:
            left_img: Left eye panorama (numpy array)
            right_img: Right eye panorama (numpy array)
            
        Returns:
            CuPy array with VR360 TB format (left on top, right on bottom)
        """
        in_height, in_width = left_img.shape[:2]
        
        # Standard equirectangular aspect ratio is 2:1
        out_width = in_width
        out_height = in_width // 2
        
        # Create black padded equirectangular images for each eye
        left_equirect = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        right_equirect = np.zeros((out_height, out_width, 3), dtype=np.uint8)
        
        # Center the source images vertically
        start_y = (out_height - in_height) // 2
        end_y = start_y + in_height
        
        left_equirect[start_y:end_y, :] = left_img
        right_equirect[start_y:end_y, :] = right_img
        
        # Stack vertically: left on top, right on bottom
        vr360 = np.vstack([left_equirect, right_equirect])
        
        # Transfer to GPU
        return cp.asarray(vr360)
    
    @property
    def image(self) -> cp.ndarray:
        """
        Get the VR360 top-bottom stereo image as a CuPy GPU array.
        
        Returns:
            CuPy array of shape (H*2, W, 3) with dtype uint8
        """
        return self._vr360_image
    
    @property
    def has_audio(self) -> bool:
        """Check if the panorama contains audio."""
        return self._compressed_audio is not None
    
    @property
    def compressed_audio(self) -> Optional[bytes]:
        """Get the compressed audio data (MP4/AAC format)."""
        return self._compressed_audio
    
    def audio_duration(self) -> Optional[float]:
        """
        Get the audio duration in seconds without decompressing.
        
        Returns:
            Duration in seconds, or None if no audio is present
        """
        if not self.has_audio:
            return None
        
        # Use ffprobe to get duration from compressed audio in memory
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            '-i', 'pipe:0'
        ]
        
        try:
            pipe = sp.Popen(
                command,
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.DEVNULL
            )
            
            output, _ = pipe.communicate(input=self._compressed_audio)
            duration_str = output.decode('utf-8').strip()
            return float(duration_str)
        except (ValueError, sp.SubprocessError):
            return None

    def decompress_audio(self, sr: int = 48000, channels: int = 2) -> Optional[np.ndarray]:
        """
        Decompress audio to float32 numpy array using existing AudioLoad.
        
        Args:
            sr: Sample rate (default: 48000 Hz)
            channels: Number of audio channels (default: 2 for stereo)
            
        Returns:
            Numpy array of shape (samples, channels) with dtype float32,
            or None if no audio is present
        """
        if not self.has_audio:
            return None
        
        import tempfile
        import os
        from AudioLoad import load_audio
        
        # Write compressed audio to temporary file for AudioLoad
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            tmp.write(self._compressed_audio)
            tmp_path = tmp.name
        
        try:
            return load_audio(tmp_path, sr=sr, channels=channels)
        finally:
            os.unlink(tmp_path)

    def save_vr360(self, output_path: str):
        """
        Save the VR360 top-bottom image to disk.
        
        Args:
            output_path: Output file path (.jpg, .png, etc.)
        """
        # Transfer from GPU to CPU
        img_cpu = cp.asnumpy(self._vr360_image)
        Image.fromarray(img_cpu).save(output_path, quality=95)
    
    def save_audio(self, output_path: str):
        """
        Save the compressed audio to disk.
        
        Args:
            output_path: Output file path (typically .mp4 or .m4a)
        """
        if not self.has_audio:
            raise ValueError("No audio data to save")
        
        with open(output_path, 'wb') as f:
            f.write(self._compressed_audio)

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python CardboardPanorama.py <cardboard_image.jpg>")
        sys.exit(1)
    
    # Load panorama
    pano = CardboardDecoder(sys.argv[1])
    
    print(f"Image shape: {pano.image.shape}")
    print(f"Image dtype: {pano.image.dtype}")
    print(f"Has audio: {pano.has_audio}")
    
    # Save VR360 version
    pano.save_vr360("output_vr360.jpg")
    print("Saved VR360 image")
    
    # Decompress and analyze audio
    if pano.has_audio:

        duration = pano.audio_duration()
        if duration:
            print(f"Audio duration: {duration:.2f} seconds")

        audio = pano.decompress_audio()
        print(f"Audio shape: {audio.shape}")
        print(f"Audio duration: {audio.shape[0] / 48000:.2f} seconds")
        
        # Optionally save audio
        pano.save_audio("output_audio.mp4")
        print("Saved audio")
