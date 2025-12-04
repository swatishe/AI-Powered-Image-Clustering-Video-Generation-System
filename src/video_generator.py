"""
Video generation module for creating slideshows from image clusters
@author sshende
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
from tqdm import tqdm


class VideoGenerator:
    """
    Generate slideshow videos from image sequences with transitions
    """
    
    def __init__(
        self,
        output_dir: str = "output/videos",
        fps: int = 2,
        resolution: tuple = (1280, 720),  # Changed to more compatible resolution
        transition_frames: int = 15
    ):
        """
        Initialize video generator
        
        Args:
            output_dir: Directory to save videos
            fps: Frames per second
            resolution: Output video resolution (width, height)
            transition_frames: Number of frames for crossfade transition
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fps = fps
        self.resolution = resolution
        self.transition_frames = transition_frames
        
        print(f"🎥 Video Generator initialized")
        print(f"   Output: {self.output_dir}")
        print(f"   Resolution: {resolution[0]}x{resolution[1]}")
        print(f"   FPS: {fps}")
    
    def create_slideshow(
        self,
        image_paths: List[str],
        output_name: str = "slideshow.mp4",
        music_path: str = None,
        max_images: int = 50
    ) -> str:
        """
        Create a slideshow video from images
        
        Args:
            image_paths: List of image file paths
            output_name: Output video filename
            music_path: Path to background music file (optional)
            max_images: Maximum number of images to include
            
        Returns:
            Path to generated video file
        """
        if len(image_paths) > max_images:
            print(f"  Limiting to {max_images} images (from {len(image_paths)})")
            image_paths = image_paths[:max_images]
        
        if len(image_paths) == 0:
            print(" Error: No images to process")
            return None
        
        output_path = self.output_dir / output_name
        temp_output_path = self.output_dir / f"temp_{output_name}"
        
        print(f"\n Creating slideshow video...")
        print(f"   Images: {len(image_paths)}")
        print(f"   Output: {output_path}")
        
        # Initialize video writer with H264 codec
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H264 codec
        video_writer = cv2.VideoWriter(
            str(temp_output_path),
            fourcc,
            self.fps,
            self.resolution
        )
        
        if not video_writer.isOpened():
            print("  Trying alternative codec (mp4v)...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(temp_output_path),
                fourcc,
                self.fps,
                self.resolution
            )
        
        # Pre-load all frames to avoid repeated I/O
        print(" Loading images...")
        frames = []
        valid_paths = []
        
        for img_path in tqdm(image_paths, desc="Loading images"):
            try:
                frame = self._prepare_frame(img_path)
                if frame is not None:
                    frames.append(frame)
                    valid_paths.append(img_path)
            except Exception as e:
                print(f"  Error loading {img_path}: {e}")
        
        if len(frames) == 0:
            print(" Error: No valid frames loaded")
            video_writer.release()
            return None
        
        print(f" Loaded {len(frames)} valid frames")
        
        # Generate video with transitions
        print(" Generating video frames...")
        
        for i in tqdm(range(len(frames)), desc="Writing video"):
            current_frame = frames[i]
            
            # Add crossfade transition (except for first image)
            if i > 0 and self.transition_frames > 0:
                prev_frame = frames[i-1]
                
                for t in range(self.transition_frames):
                    alpha = t / self.transition_frames
                    # Blend frames
                    blended = cv2.addWeighted(
                        prev_frame.astype(np.float32), 1 - alpha,
                        current_frame.astype(np.float32), alpha,
                        0
                    ).astype(np.uint8)
                    
                    video_writer.write(blended)
            
            # Write main frame (hold for 1 second)
            frames_to_hold = max(1, self.fps)  # Hold each image for 1 second
            for _ in range(frames_to_hold):
                video_writer.write(current_frame)
        
        # Release video writer
        video_writer.release()
        
        print(f" Video frames written successfully")
        
        # Add music if provided
        if music_path and Path(music_path).exists():
            print(f"\n Adding background music...")
            final_output = self._add_audio(temp_output_path, music_path)
            
            if final_output != temp_output_path:
                # Remove temp file if audio was added successfully
                if temp_output_path.exists():
                    temp_output_path.unlink()
                return str(final_output)
            else:
                # Rename temp file to final output
                temp_output_path.rename(output_path)
                return str(output_path)
        else:
            if music_path:
                print(f" Music file not found: {music_path}")
            # Rename temp file to final output
            temp_output_path.rename(output_path)
            return str(output_path)
    
    def _prepare_frame(self, image_path: str) -> np.ndarray:
        """
        Load and prepare image frame with proper color handling
        
        Args:
            image_path: Path to image file
            
        Returns:
            Processed frame as numpy array in BGR format for OpenCV
        """
        try:
            # Load image using PIL (handles various formats better)
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get original dimensions
            orig_width, orig_height = img.size
            target_width, target_height = self.resolution
            
            # Calculate scaling to fit within target resolution
            scale = min(target_width / orig_width, target_height / orig_height)
            
            # Calculate new dimensions
            new_width = int(orig_width * scale)
            new_height = int(orig_height * scale)
            
            # Resize image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            # Create black canvas
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Calculate position to center image
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            
            # Place image on canvas
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = img_array
            
            # Convert RGB to BGR for OpenCV (CRITICAL!)
            canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            
            return canvas_bgr
            
        except Exception as e:
            print(f"  Error preparing frame from {image_path}: {e}")
            return None
    
    def _add_audio(self, video_path: Path, audio_path: str) -> Path:
        """
        Add background music to video using moviepy or ffmpeg
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            
        Returns:
            Path to video with audio
        """
        output_path = video_path.parent / video_path.name.replace("temp_", "")
        
        # Try using moviepy first
        try:
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            print(f"   Using MoviePy to add audio...")
            
            # Load video and audio
            video = VideoFileClip(str(video_path))
            audio = AudioFileClip(audio_path)
            
            # Loop or trim audio to match video duration
            if audio.duration < video.duration:
                # Calculate number of loops needed
                n_loops = int(np.ceil(video.duration / audio.duration))
                from moviepy.audio.AudioClip import concatenate_audioclips
                audio = concatenate_audioclips([audio] * n_loops)
            
            # Trim audio to video length
            audio = audio.subclip(0, min(audio.duration, video.duration))
            
            # Set audio to video
            final_video = video.set_audio(audio)
            
            # Write output with proper codec
            final_video.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                fps=self.fps,
                preset='medium',
                logger=None,
                threads=4
            )
            
            # Cleanup
            video.close()
            audio.close()
            final_video.close()
            
            print(f" Music added successfully with MoviePy")
            return output_path
            
        except ImportError:
            print(f" MoviePy not available, trying ffmpeg...")
        except Exception as e:
            print(f"  MoviePy error: {e}, trying ffmpeg...")
        
        # Try using ffmpeg command line
        try:
            import subprocess
            
            cmd = [
                'ffmpeg',
                '-i', str(video_path),
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                '-y',  # Overwrite output
                str(output_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f" Music added successfully with ffmpeg")
                return output_path
            else:
                print(f" ffmpeg error: {result.stderr}")
        
        except FileNotFoundError:
            print(f" ffmpeg not found in system PATH")
        except Exception as e:
            print(f"  Error using ffmpeg: {e}")
        
        # If all methods fail, return original video
        print(f"  Could not add audio, returning video without music")
        return video_path
    
    def sort_by_similarity(
        self,
        image_paths: List[str],
        embeddings: np.ndarray
    ) -> List[str]:
        """
        Sort images by similarity using greedy nearest neighbor
        
        Args:
            image_paths: List of image paths
            embeddings: Feature embeddings for images
            
        Returns:
            Sorted list of image paths
        """
        print(f"\n Sorting {len(image_paths)} images by similarity...")
        
        if len(image_paths) <= 1:
            return image_paths
        
        # Start with first image
        sorted_indices = [0]
        remaining = set(range(1, len(embeddings)))
        
        # Greedy nearest neighbor
        while remaining:
            current_embedding = embeddings[sorted_indices[-1]]
            
            # Find nearest neighbor among remaining
            min_dist = float('inf')
            nearest_idx = None
            
            for idx in remaining:
                dist = np.linalg.norm(current_embedding - embeddings[idx])
                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = idx
            
            sorted_indices.append(nearest_idx)
            remaining.remove(nearest_idx)
        
        sorted_paths = [image_paths[i] for i in sorted_indices]
        
        print(f" Images sorted by similarity")
        return sorted_paths
    
    def create_thumbnail_grid(
        self,
        image_paths: List[str],
        output_path: str,
        grid_size: tuple = (5, 5)
    ):
        """
        Create a thumbnail grid visualization
        
        Args:
            image_paths: List of image paths
            output_path: Path to save grid image
            grid_size: Grid dimensions (rows, cols)
        """
        rows, cols = grid_size
        n_images = min(len(image_paths), rows * cols)
        
        thumb_size = 200
        grid_img = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
        
        for i in range(n_images):
            row = i // cols
            col = i % cols
            
            try:
                img = Image.open(image_paths[i]).convert('RGB')
                img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                img_array = np.array(img)
                
                # Center thumbnail in cell
                y_offset = (thumb_size - img_array.shape[0]) // 2
                x_offset = (thumb_size - img_array.shape[1]) // 2
                
                grid_img[
                    row * thumb_size + y_offset:row * thumb_size + y_offset + img_array.shape[0],
                    col * thumb_size + x_offset:col * thumb_size + x_offset + img_array.shape[1]
                ] = img_array
                
            except Exception:
                pass
        
        # Save grid
        grid_pil = Image.fromarray(grid_img)
        grid_pil.save(output_path)
        print(f"Thumbnail grid saved to: {output_path}")