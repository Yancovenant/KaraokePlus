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
        env.librosa, env.torch, env.numpy, env.scipy
        import librosa, torch, numpy as np
        import matplotlib.pyplot as plt
        from scipy.signal import find_peaks
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy().squeeze()
        elif not isinstance(audio, np.ndarray):
            audio, lsr = librosa.load(audio, sr=None)
            sr = lsr
            logger.debug(f"Changing sr value from {sr} to {lsr}")
            # assert lsr == sr, f"Given samplerate and loaded sample rate is different: {lsr} | {sr}" 
        with MainProgress(total=1, desc=f"Processing audio: {len(audio)} samples, {sr}Hz, {len(audio)/sr:.2f}") as main_bar:
            main_bar.pbar.set_description("Computing RMS")
            precision_ms = 100 # 100ms
            hop_length = int(sr / 1000) * precision_ms
            frame_length = int(hop_length * 1.5)
            distance = max(1, int((precision_ms * sr) / (1000 * hop_length)))
            rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            inverted_rms = rms * -1
            all_valleys, _ = find_peaks(inverted_rms, distance=distance, prominence=0.01)
            
            # precision_ms = 100
            # frame_length=512
            # hop_length = int(sr / 1000) * precision_ms
            # distance = max(1, int((precision_ms * sr) / (1000 * hop_length)))
            # frame_duration = hop_length / sr
            # rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
            # inverted_rms = rms * -1

            # all_valleys, _ = find_peaks(inverted_rms, distance=distance, prominence=0.01)
            # candidate_frames = np.concatenate(([0], all_valleys, [len(rms)]))
            # segment_mean_threshold = 0.05 

            # valid_segments = []
            # current_start_frame = None
            # current_end_frame = None

            # for i in range(len(candidate_frames) - 1):
            #     start_frame = candidate_frames[i]
            #     end_frame = candidate_frames[i+1]
            #     segment_duration = (end_frame - start_frame) * frame_duration
            #     # Calculate the average energy of this specific chunk
            #     chunk_mean_energy = np.mean(rms[start_frame:end_frame])
                
            #     if chunk_mean_energy >= segment_mean_threshold:
            #         if current_start_frame is None:
            #             current_start_frame = start_frame
            #         current_end_frame = end_frame 
            #     else:
            #         if current_start_frame is not None:
            #             if segment_duration < 0.8: # if its less than 800ms gap we wanna use it?
            #                 current_end_frame = end_frame
            #             valid_segments.append((current_start_frame, current_end_frame))
            #             current_start_frame = None # Reset for the next vocal phrase

            # # Catch any active segment that touched the very end of the file
            # if current_start_frame is not None:
            #     valid_segments.append((current_start_frame, current_end_frame))


            # # --- VISUALIZATION ---
            # fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
            # time_axis = np.arange(len(audio)) / sr

            # # Draw the baseline in grey (this represents the rejected/silent chunks)
            # axes[0].plot(time_axis, audio, linewidth=0.5, color='darkgray', alpha=0.5)
            # librosa.display.waveshow(audio, sr=sr, ax=axes[1], color='darkgray', alpha=0.5)

            # # Alternating color palette for the final merged segments
            # segment_colors = ['steelblue', 'limegreen'] 

            # for i, (start_frame, end_frame) in enumerate(valid_segments):
            #     # Convert the validated frame boundaries back to raw samples and seconds
            #     start_sample = start_frame * hop_length
            #     end_sample = min(end_frame * hop_length, len(audio)) # Prevent index out of bounds
                
            #     start_t = start_sample / sr
            #     end_t = end_sample / sr
                
            #     # Plot the kept segments over the grey baseline
            #     axes[0].plot(
            #         time_axis[start_sample:end_sample], 
            #         audio[start_sample:end_sample], 
            #         linewidth=0.5, 
            #         color=segment_colors[i % 2], 
            #         alpha=1.0
            #     )
                
            #     # Shade the active blocks 
            #     axes[0].axvspan(start_t, end_t, color='black', alpha=0.05)
            #     axes[1].axvspan(start_t, end_t, color='black', alpha=0.05)
                
            #     # Draw red boundary lines strictly at the edges of the merged segments
            #     axes[0].axvline(x=start_t, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            #     axes[0].axvline(x=end_t, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            #     axes[1].axvline(x=start_t, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            #     axes[1].axvline(x=end_t, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

            # # Draw the visual threshold lines referencing your image
            # axes[0].axhline(y=segment_mean_threshold, color='red', linestyle='-', linewidth=2, alpha=0.8)
            # axes[0].axhline(y=-segment_mean_threshold, color='red', linestyle='-', linewidth=2, alpha=0.8)

            # axes[0].set_title(f"Raw Waveform (Merged Segments | Mean Threshold: {segment_mean_threshold})")
            # axes[0].set_ylabel("Amplitude")
            # axes[1].set(title='Slower Version $X_1$')
            # axes[1].label_outer()
            # main_bar.update(1)
            # logger.debug(f"RMS computed: {len(rms_db)} frames")
            # logger.debug(f"RMS range: {rms_db.min():.1f}dB to {rms_db.max():.1f}dB")
            plt.tight_layout()
            if self.visual:
                plt.show()
            plt.close()
        
            return []
