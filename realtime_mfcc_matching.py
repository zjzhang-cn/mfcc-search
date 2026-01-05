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

class RealtimeAudioMatcher:
    def __init__(self, reference_file, sr=16000, n_mfcc=40, window_duration=1.0, threshold=0.95):
        """
        Initialize the real-time audio matcher.
        
        Args:
            reference_file: Path to reference audio file
            sr: Sample rate (default: 16000 Hz)
            n_mfcc: Number of MFCC coefficients (default: 40)
            window_duration: Window duration in seconds (default: 1.0)
            threshold: Similarity threshold for saving segments (default: 0.95)
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
        
        # Create output directory for audio segments
        self.output_dir = 'high_similarity_segments'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.segment_counter = 0
        
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
            
            # Record and save high similarity matches (>95%)
            if similarity > self.similarity_threshold:
                saved_file = self.save_audio_segment(audio_data, similarity, best_time)
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
    
    def start_capturing(self, duration=None, device=None):
        """
        Start real-time microphone capture.
        
        Args:
            duration: Capture duration in seconds (None for infinite)
            device: Audio device index (None for default)
        """
        print(f"\nStarting real-time audio capture...")
        print(f"Window size: {self.window_duration}s")
        print(f"Press Ctrl+C to stop\n")
        
        self.running = True
        
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
        threshold=args.threshold
    )
    
    # Start capturing
    matcher.start_capturing(
        duration=args.duration,
        device=args.device
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
                print(f"   File: {record['file']}")
        print(f"{'='*70}")
        print(f"Total matches found: {len(matcher.high_similarity_records)}")
        print(f"Output directory: {matcher.output_dir}")
    else:
        print(f"\n\nNo matches with similarity > {threshold_percent:.1f}% found")


if __name__ == '__main__':
    main()
