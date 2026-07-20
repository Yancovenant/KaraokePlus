from __future__ import annotations

import logging

from pathlib import Path
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, TypeAlias

from kplus.tools.progress import MainProgress
from kplus.environment import env

if TYPE_CHECKING:
    import numpy as np
    import torch
    AudioType : TypeAlias = "torch.Tensor | np.ndarray | str"


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AudioSegment:
    start: float
    end: float
    text: str = ""
    language: str = ""
    pitch_present: bool = False
    confidence: float = 0.0
    is_refined: bool = False


class AAD:
    """ Audio Activity Detection via RMS/DB 
    """
    def __init__(self, visualize: bool = False):
        self.visual = visualize
        env.matplotlib, env.torchcrepe
        import matplotlib.pyplot as plt
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (20, 12)
        plt.rcParams['font.size'] = 10
    
    def visualize(self, audio: np.ndarray, sr: int, segments: List[AudioSegment],
                  output_path: Path, show: bool = False):
        env.librosa, env.numpy
        import matplotlib.pyplot as plt
        import librosa, numpy as np
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
        
        # Plot waveform
        time_axis = np.arange(len(audio)) / sr
        ax1.plot(time_axis, audio, linewidth=0.5, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Waveform with Detected Segments')
        
        # Highlight segments
        for seg in segments:
            ax1.axvspan(seg.start, seg.end, alpha=0.3, color='green', label='Detected segment')
        
        # Plot RMS energy
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        rms_time = np.arange(len(rms_db)) * 512 / sr
        
        ax2.plot(rms_time, rms_db, linewidth=1.5, color='coral')
        ax2.axhline(y=-40, color='red', linestyle='--', label='Threshold (-40dB)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RMS (dB)')
        ax2.set_title('RMS Energy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"visualization saved: {output_path}")
        
        if show:
            plt.show()
        plt.close()

    def detect(self, audio: AudioType, sr: int, 
               frame_length: int = 2048, hop_length: int = 512,
               rms_threshold_db: float = -40,
               min_segment_duration: float = 0.5) -> List[AudioSegment]:
        """ Detect audio activity using RMS energy
        
            Args:
                audio: Audio signal
                sr: Sample rate
                frame_length: FFT window size
                hop_length: Hop length between frames
                rms_threshold_db: RMS threshold in dB
                min_segment_duration: Minimum segment duration in seconds
                
            Returns:
                List of audio segments
        """
        env.librosa, env.torch, env.numpy
        import librosa, torch, numpy as np
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy().squeeze()
        elif not isinstance(audio, np.ndarray):
            audio, lsr = librosa.load(audio, sr=None)
            sr = lsr
            logger.debug(f"Changing sr value from {sr} to {lsr}")
            # assert lsr == sr, f"Given samplerate and loaded sample rate is different: {lsr} | {sr}" 
        with MainProgress(total=0, desc=f"Processing audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}") as main_bar:
            main_bar.pbar.set_description("Computing RMS")
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            main_bar.update(1)
            logger.debug(f"RMS computed: {len(rms_db)} frames")
            logger.debug(f"RMS range: {rms_db.min():.1f}dB to {rms_db.max():.1f}dB")

            #
            active_frames = rms_db > rms_threshold_db
            segments = []
            in_segment = False
            segment_start = 0
            for i, is_active in enumerate(active_frames):
                frame_time = i * hop_length / sr
                
                if is_active and not in_segment:
                    # Start of new segment
                    segment_start = frame_time
                    in_segment = True
                elif not is_active and in_segment:
                    # End of segment
                    segment_end = frame_time
                    duration = segment_end - segment_start
                    
                    if duration >= min_segment_duration:
                        segments.append(AudioSegment(
                            start=segment_start,
                            end=segment_end,
                            confidence=1.0,
                        ))
                        logger.debug(f"Segment detected: {segment_start:.3f}s - {segment_end:.3f}s ({duration:.3f}s)")
                    
                    in_segment = False
            if in_segment:
                segment_end = len(audio) / sr
                duration = segment_end - segment_start
                if duration >= min_segment_duration:
                    segments.append(AudioSegment(
                        start=segment_start,
                        end=segment_end,
                        confidence=1.0,
                    ))
                    logger.debug(f"Final segment: {segment_start:.3f}s - {segment_end:.3f}s ({duration:.3f}s)")
            
            logger.info(f"{len(segments)} segments found")
            if self.visual:
                self.visualize(audio, sr, segments, "testrms2.jpg", True)
            
            # CREPE PITCH
            logger.info(f"Processing {len(segments)} segments with pitch tracking")
            hop_length: int = 160
            confidence_threshold: float = 0.5
            min_segment_duration: float = 1.0
            max_gap: float = 0.2
            audio_ts = torch.from_numpy(audio).unsqueeze(0).to(env.device)
            import torchcrepe
            pitch, periodicity = torchcrepe.predict(
                audio_ts,
                sample_rate=sr,
                hop_length=hop_length,
                fmin=50,  # Minimum frequency (bass)
                fmax=1000,  # Maximum frequency (soprano)
                model='full',
                batch_size=64,#2048,
                device=env.device,
                return_periodicity=True,
                pad=True
            )
            pitch = pitch.squeeze().cpu().numpy()
            periodicity = periodicity.squeeze().cpu().numpy()
            
            self.logger.debug(f"Pitch tracking complete: {len(pitch)} frames")

            refined_segments = []
            
            for seg in segments:
                # Get pitch frames within segment
                start_frame = int(seg.start * sr / hop_length)
                end_frame = int(seg.end * sr / hop_length)
                
                start_frame = max(0, start_frame)
                end_frame = min(len(pitch), end_frame)
                
                seg_pitch = pitch[start_frame:end_frame]
                seg_conf = periodicity[start_frame:end_frame]
                
                # Check if segment has pitch
                has_pitch = np.mean(seg_conf > confidence_threshold) > 0.3
                
                if has_pitch:
                    # Find voiced frames within segment
                    voiced_mask = seg_conf > confidence_threshold
                    
                    # Find contiguous voiced regions
                    voiced_regions = []
                    in_voiced = False
                    voiced_start = 0
                    
                    for i, is_voiced in enumerate(voiced_mask):
                        frame_time = (start_frame + i) * hop_length / sr
                        
                        if is_voiced and not in_voiced:
                            voiced_start = frame_time
                            in_voiced = True
                        elif not is_voiced and in_voiced:
                            voiced_end = frame_time
                            if voiced_end - voiced_start >= 0.05:  # 50ms minimum
                                voiced_regions.append((voiced_start, voiced_end))
                            in_voiced = False
                    
                    # Handle last voiced region
                    if in_voiced:
                        voiced_end = seg.end
                        if voiced_end - voiced_start >= 0.05:
                            voiced_regions.append((voiced_start, voiced_end))
                    
                    # Merge contiguous regions with small gaps
                    merged_regions = self.merge_contiguous_regions(
                        voiced_regions, 
                        max_gap=max_gap
                    )
                    
                    # Filter by minimum duration
                    for v_start, v_end in merged_regions:
                        duration = v_end - v_start
                        if duration >= min_segment_duration:
                            refined_segments.append(AudioSegment(
                                start=v_start,
                                end=v_end,
                                pitch_present=True,
                                confidence=float(np.mean(seg_conf[(seg_conf > confidence_threshold)])),
                                is_refined=True
                            ))
                            self.logger.debug(f"Refined segment: {v_start:.3f}s - {v_end:.3f}s ({duration:.3f}s)")
                        else:
                            self.logger.debug(f"Discarded short segment: {v_start:.3f}s - {v_end:.3f}s ({duration:.3f}s)")
                else:
                    # Keep original segment if no clear pitch
                    refined_segments.append(seg)
                    self.logger.debug(f"Kept original segment: {seg.start:.3f}s - {seg.end:.3f}s")
            
            # Log statistics
            orig_count = len(segments)
            refined_count = len(refined_segments)
            self.logger.info(f"{orig_count} → {refined_count} segments "
                            f"({refined_count/orig_count:.1f}x)")
            if self.visual:
                self.visualize_refine(audio, sr, refined_segments,
                                      pitch, periodicity, hop_length, "testrefine2.jpg", True)
            
           
        return refined_segments

    def visualize_refine(self, audio: np.ndarray, sr: int, segments: List[AudioSegment],
                  pitch: np.ndarray, periodicity: np.ndarray, hop_length: int,
                  output_path: Path, show: bool = False):
        """Visualize pitch refinement results with duration control"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), sharex=True)
        
        time_axis = np.arange(len(audio)) / sr
        
        # Plot waveform
        ax1.plot(time_axis, audio, linewidth=0.5, color='steelblue', alpha=0.7)
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Waveform with Pitch-Based Segments (Duration ≥1s)')
        
        # Highlight refined segments
        for seg in segments:
            if seg.is_refined:
                color = 'green'
                label = 'Refined segment (≥1s)'
            else:
                color = 'orange'
                label = 'Original segment'
            ax1.axvspan(seg.start, seg.end, alpha=0.3, color=color)
        
        # Add legend
        green_patch = mpatches.Patch(color='green', alpha=0.3, label='Refined segment (≥1s)')
        orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='Original segment')
        ax1.legend(handles=[green_patch, orange_patch], loc='upper right')
        
        # Plot pitch contour
        pitch_time = np.arange(len(pitch)) * hop_length / sr
        ax2.plot(pitch_time, pitch, linewidth=2, color='red', label='Pitch (Hz)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Pitch Contour')
        ax2.legend()
        
        # Plot periodicity/confidence
        ax3.plot(pitch_time, periodicity, linewidth=2, color='purple', label='Confidence')
        ax3.axhline(y=0.5, color='red', linestyle='--', label='Threshold')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Confidence')
        ax3.set_title('Pitch Confidence (Periodicity)')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.logger.info(f"Refine visualization saved: {output_path}")
        
        if show:
            plt.show()
        plt.close()

    def merge_contiguous_regions(self, regions: List[Tuple[float, float]], 
                                max_gap: float = 0.2) -> List[Tuple[float, float]]:
        """Merge regions that are within max_gap of each other"""
        if not regions:
            return []
        
        regions = sorted(regions, key=lambda x: x[0])
        merged = []
        current_start, current_end = regions[0]
        
        for i in range(1, len(regions)):
            next_start, next_end = regions[i]
            if next_start - current_end <= max_gap:
                current_end = max(current_end, next_end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        
        merged.append((current_start, current_end))
        return merged