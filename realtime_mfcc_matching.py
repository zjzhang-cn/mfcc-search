import argparse
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from scipy.spatial.distance import cosine
from collections import deque
import threading
import time
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class RealtimeAudioMatcher:
    def __init__(self, reference_file, sr=16000, n_mfcc=40, window_duration=1.0, threshold=0.95, save_files=False):
        """
        Initialize the real-time audio matcher.
        
        Args:
            reference_file: Path to reference audio file
            sr: Sample rate (default: 16000 Hz)
            n_mfcc: Number of MFCC coefficients (default: 40)
            window_duration: Window duration in seconds (default: 1.0)
            threshold: Similarity threshold for saving segments (default: 0.95)
            save_files: Whether to save audio files (default: False)
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.window_duration = window_duration
        self.n_fft = 512
        self.hop_length = int(sr * 0.01)
        self.win_length = int(sr * 0.025)
        
        # Load and compute reference MFCC
        print(f"Loading reference audio: {reference_file}")
        y, sr_ref = librosa.load(reference_file, sr=self.sr)
        self.reference_mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        print(f"Reference MFCC shape: {self.reference_mfcc.shape}")
        
        # Buffer for incoming audio
        self.window_size = int(sr * window_duration)
        self.audio_buffer = deque(maxlen=self.window_size)
        
        # Results storage
        self.last_similarity = None
        self.last_best_time = None
        self.running = False
        self.high_similarity_records = []  # Record high similarity matches
        self.similarity_threshold = threshold  # Configurable threshold
        self.save_files = save_files  # Whether to save audio files
        
        # Create output directory for audio segments
        self.output_dir = 'high_similarity_segments'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.segment_counter = 0
        
        # Visualization data
        self.similarity_history = deque(maxlen=200)  # Keep last 200 data points
        self.time_history = deque(maxlen=200)
        self.start_time = None
        self.enable_plot = False
        self.fig = None
        self.ax = None
        self.line = None
        self.match_markers = []  # Store match position markers on reference MFCC
        
    def compute_mfcc(self, audio):
        """Compute MFCC from audio signal"""
        if len(audio) < self.n_fft:
            return None
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
    
    def normalize_mfcc(self, mfcc):
        """Normalize MFCC features"""
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True) + 1e-8
        return (mfcc - mean) / std
    
    def compute_stats_features(self, mfcc):
        """Compute statistical features from MFCC"""
        return np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.max(mfcc, axis=1),
            np.min(mfcc, axis=1),
        ])
    
    def match_mfcc(self, query_mfcc):
        """
        Match query MFCC against reference MFCC using sliding window.
        
        Returns:
            similarity: Best match similarity score (0-1)
            best_position: Best match position in reference (seconds)
        """
        if query_mfcc is None or query_mfcc.shape[1] < 2:
            return 0.0, 0.0
        
        query_frames = query_mfcc.shape[1]
        target_frames = self.reference_mfcc.shape[1]
        
        if query_frames > target_frames:
            return 0.0, 0.0
        
        # Normalize query
        query_norm = self.normalize_mfcc(query_mfcc)
        query_flat = query_norm.flatten()
        query_stats = self.compute_stats_features(query_mfcc)
        
        best_similarity = -1.0
        best_position = 0
        
        # Sliding window matching
        for i in range(target_frames - query_frames + 1):
            window = self.reference_mfcc[:, i:i+query_frames]
            
            window_norm = self.normalize_mfcc(window)
            window_flat = window_norm.flatten()
            window_stats = self.compute_stats_features(window)
            
            # Compute similarities
            cos_dist_norm = cosine(window_flat, query_flat)
            sim_norm = 1 - cos_dist_norm
            
            cos_dist_stats = cosine(window_stats, query_stats)
            sim_stats = 1 - cos_dist_stats
            
            # Weighted combination
            sim = 0.7 * sim_norm + 0.3 * sim_stats
            
            if sim > best_similarity:
                best_similarity = sim
                best_position = i
        
        best_time = best_position * self.hop_length / self.sr
        return best_similarity, best_time
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback for real-time capture"""
        if status:
            print(f"Audio callback status: {status}")
        
        audio_chunk = indata[:, 0]
        self.audio_buffer.extend(audio_chunk)
    
    def setup_visualization(self):
        """Setup matplotlib for real-time visualization"""
        plt.ion()  # Turn on interactive mode
        self.fig = plt.figure(figsize=(14, 10))
        
        # Create grid layout: 3 rows (similarity plot, captured MFCC, reference MFCC)
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
        
        # Similarity plot
        self.ax = self.fig.add_subplot(gs[0])
        self.line, = self.ax.plot([], [], 'b-', linewidth=1.5, label='Similarity')
        self.threshold_line = self.ax.axhline(y=self.similarity_threshold, color='r', 
                                              linestyle='--', linewidth=2, 
                                              label=f'Threshold ({self.similarity_threshold:.2f})')
        self.ax.set_xlim(0, 10)  # Show last 10 seconds
        self.ax.set_ylim(0, 1.0)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Similarity')
        self.ax.set_title('Real-time Audio Similarity')
        self.ax.legend(loc='upper right')
        self.ax.grid(True, alpha=0.3)
        
        # Captured MFCC plot
        self.ax_mfcc_captured = self.fig.add_subplot(gs[1])
        self.mfcc_captured_img = None
        self.ax_mfcc_captured.set_xlabel('Time frames')
        self.ax_mfcc_captured.set_ylabel('MFCC coefficient')
        self.ax_mfcc_captured.set_title('Captured MFCC (High similarity segments)')
        
        # Reference MFCC plot
        self.ax_mfcc_ref = self.fig.add_subplot(gs[2])
        ref_img = self.ax_mfcc_ref.imshow(self.reference_mfcc, aspect='auto', origin='lower', cmap='viridis')
        self.fig.colorbar(ref_img, ax=self.ax_mfcc_ref, label='Amplitude')
        self.ax_mfcc_ref.set_xlabel('Time frames')
        self.ax_mfcc_ref.set_ylabel('MFCC coefficient')
        self.ax_mfcc_ref.set_title('Reference MFCC')
        
        plt.tight_layout()
        self.fig.show()
        self.enable_plot = True
    
    def update_visualization(self):
        """Update the real-time plot"""
        if not self.enable_plot or self.fig is None:
            return
        
        if len(self.similarity_history) > 0:
            times = list(self.time_history)
            sims = list(self.similarity_history)
            
            self.line.set_data(times, sims)
            
            # Auto-adjust x-axis to show last 10 seconds
            if len(times) > 0:
                max_time = max(times)
                self.ax.set_xlim(max(0, max_time - 10), max_time + 1)
            
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except:
                pass
    
    def update_mfcc_display(self, mfcc, similarity, best_position, query_frames):
        """Update MFCC display in the visualization window"""
        if not self.enable_plot or self.fig is None:
            return
        
        try:
            # Clear previous MFCC image if exists
            if self.mfcc_captured_img is not None:
                self.mfcc_captured_img.remove()
            
            # Display new MFCC
            self.mfcc_captured_img = self.ax_mfcc_captured.imshow(
                mfcc, aspect='auto', origin='lower', cmap='viridis'
            )
            self.ax_mfcc_captured.set_title(f'Captured MFCC (Similarity: {similarity:.4f})')
            
            # Add colorbar if not exists
            if not hasattr(self, 'mfcc_captured_cbar'):
                self.mfcc_captured_cbar = self.fig.colorbar(
                    self.mfcc_captured_img, ax=self.ax_mfcc_captured, label='Amplitude'
                )
            
            # Clear previous match markers on reference MFCC
            for marker in self.match_markers:
                marker.remove()
            self.match_markers.clear()
            
            # Add match position marker on reference MFCC
            match_start = best_position
            match_end = best_position + query_frames
            
            # Draw rectangle to highlight the match region
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (match_start, 0),
                query_frames,
                self.n_mfcc,
                linewidth=2,
                edgecolor='red',
                facecolor='none',
                linestyle='--'
            )
            self.ax_mfcc_ref.add_patch(rect)
            self.match_markers.append(rect)
            
            # Add vertical lines for better visibility
            line1 = self.ax_mfcc_ref.axvline(x=match_start, color='r', linestyle='--', linewidth=2, alpha=0.7)
            line2 = self.ax_mfcc_ref.axvline(x=match_end, color='orange', linestyle='--', linewidth=2, alpha=0.7)
            self.match_markers.extend([line1, line2])
            
            # Update title with match position
            best_time_seconds = match_start * self.hop_length / self.sr
            self.ax_mfcc_ref.set_title(
                f'Reference MFCC (Match at frame {match_start}, time: {best_time_seconds:.2f}s)'
            )
            
        except Exception as e:
            print(f"  Error updating MFCC display: {e}")
    
    def save_audio_segment(self, audio_data, similarity, best_time):
        """Save audio segment to file"""
        self.segment_counter += 1
        filename = f"{self.output_dir}/segment_{self.segment_counter:03d}_sim{similarity:.4f}.wav"
        try:
            sf.write(filename, audio_data, self.sr)
            print(f"  Saved: {filename}")
            return filename
        except Exception as e:
            print(f"  Error saving file: {e}")
            return None
    
    def process_window(self):
        """Process audio window and perform matching"""
        if len(self.audio_buffer) < self.window_size:
            return
        
        audio_data = np.array(list(self.audio_buffer))
        
        # Compute MFCC for current window
        current_mfcc = self.compute_mfcc(audio_data)
        
        if current_mfcc is not None:
            similarity, best_time = self.match_mfcc(current_mfcc)
            self.last_similarity = similarity
            self.last_best_time = best_time
            
            # Add to visualization history
            if self.start_time is not None:
                elapsed = time.time() - self.start_time
                self.similarity_history.append(similarity)
                self.time_history.append(elapsed)
                self.update_visualization()
            
            # Record and save high similarity matches (>95%)
            if similarity > self.similarity_threshold:
                saved_file = None
                if self.save_files:
                    saved_file = self.save_audio_segment(audio_data, similarity, best_time)
                # Display MFCC in visualization window with match position
                query_frames = current_mfcc.shape[1]
                best_position = int(best_time * self.sr / self.hop_length)
                self.update_mfcc_display(current_mfcc, similarity, best_position, query_frames)
                self.high_similarity_records.append({
                    'similarity': similarity,
                    'best_time': best_time,
                    'timestamp': time.time(),
                    'file': saved_file
                })
                print(f"\n✓ High match found! Similarity: {similarity:.4f} at {best_time:.2f}s")
            
            # Print result
            status_bar = "=" * int(similarity * 50)
            print(f"\r[{status_bar:<50}] Similarity: {similarity:.4f} | Best match at: {best_time:.2f}s", end='')
    
    def start_capturing(self, duration=None, device=None, enable_plot=True):
        """
        Start real-time microphone capture.
        
        Args:
            duration: Capture duration in seconds (None for infinite)
            device: Audio device index (None for default)
            enable_plot: Enable real-time plotting (default: True)
        """
        print(f"\nStarting real-time audio capture...")
        print(f"Window size: {self.window_duration}s")
        print(f"Press Ctrl+C to stop\n")
        
        # Setup visualization
        if enable_plot:
            self.setup_visualization()
        
        self.running = True
        self.start_time = time.time()
        
        try:
            with sd.InputStream(
                device=device,
                samplerate=self.sr,
                channels=1,
                blocksize=4096,
                callback=self.audio_callback,
                latency='low'
            ):
                start_time = time.time()
                last_process_time = start_time
                
                while self.running:
                    # Process window every 0.5 seconds to avoid excessive computation
                    current_time = time.time()
                    if current_time - last_process_time >= 0.5:
                        self.process_window()
                        last_process_time = current_time
                    
                    # Check duration limit
                    if duration is not None and (current_time - start_time) >= duration:
                        break
                    
                    time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\nCapture stopped by user")
        finally:
            self.running = False
            if self.enable_plot and self.fig is not None:
                plt.ioff()
                plt.close(self.fig)
    
    def list_devices(self):
        """List available audio devices"""
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"  Device {i}: {device['name']}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time microphone audio matching against reference'
    )
    parser.add_argument(
        '-r', '--reference',
        required=True,
        help='Path to reference audio file'
    )
    parser.add_argument(
        '-w', '--window',
        type=float,
        default=1.0,
        help='Window duration in seconds (default: 1.0)'
    )
    parser.add_argument(
        '-d', '--duration',
        type=float,
        default=None,
        help='Total capture duration in seconds (default: infinite)'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=None,
        help='Audio device index'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.95,
        help='Similarity threshold for saving segments (default: 0.95, range: 0.0-1.0)'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Enable saving audio files (default: disabled)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable real-time visualization'
    )
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio devices and exit'
    )
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        matcher = RealtimeAudioMatcher(args.reference, window_duration=args.window)
        matcher.list_devices()
        return
    
    # Validate threshold
    if not (0.0 <= args.threshold <= 1.0):
        print(f"Error: threshold must be between 0.0 and 1.0, got {args.threshold}")
        return
    
    # Create matcher
    matcher = RealtimeAudioMatcher(
        args.reference,
        window_duration=args.window,
        threshold=args.threshold,
        save_files=args.save
    )
    
    # Start capturing
    matcher.start_capturing(
        duration=args.duration,
        device=args.device,
        enable_plot=not args.no_plot
    )
    
    # Print summary
    if matcher.last_similarity is not None:
        print(f"\n\nFinal result:")
        print(f"  Best similarity: {matcher.last_similarity:.4f}")
        print(f"  Best match position: {matcher.last_best_time:.2f}s")
    
    # Print all high similarity matches
    threshold_percent = matcher.similarity_threshold * 100
    if matcher.high_similarity_records:
        print(f"\n\n{'='*70}")
        print(f"High similarity matches (> {threshold_percent:.1f}%):")
        print(f"{'='*70}")
        for i, record in enumerate(matcher.high_similarity_records, 1):
            print(f"{i}. Similarity: {record['similarity']:.4f} | Best match at: {record['best_time']:.2f}s")
            if record['file']:
                print(f"   Audio: {record['file']}")
        print(f"{'='*70}")
        print(f"Total matches found: {len(matcher.high_similarity_records)}")
        print(f"Output directory: {matcher.output_dir}")
    else:
        print(f"\n\nNo matches with similarity > {threshold_percent:.1f}% found")


if __name__ == '__main__':
    main()
