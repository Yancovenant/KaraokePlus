import logging
import glob
import os

from pathlib import Path

from kplus.environment import env
from kplus.tools.config import config
from kplus.tools.progress import MainProgress


logger = logging.getLogger(__name__)


class VisualizeWaveform:
    def __init__(self):
        pass

    def visualize(self, input_path: Path,
                  f_length: int = 1024,
                  h_length: int = 160) -> None:
        env.librosa, env.matplotlib, env.numpy
        import librosa
        import numpy as np
        import matplotlib.pyplot as plt
        with MainProgress(total=7, desc="Plotting visual graphic...", unit="step") as main_bar:
            filename = str(Path(input_path).stem)
            main_bar.pbar.set_description(f"Loading file: {filename}")
            y, sr = librosa.load(input_path, sr=16000, mono=True) # MMS_FA Sample rate
            main_bar.update(1)
            # 5 rows for 5 different visual analyses
            fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(60, 18), sharex=True)
            fig.suptitle("Audio Separation Preset Data", fontsize=16)

            ## Waveform (Amplitude / dB)
            main_bar.pbar.set_description(f"Generating waveshow..: samplerate:{sr}")
            librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=0.7)
            axes[0].set_title(f"{filename} - Waveform", weight='bold')
            axes[0].set_ylabel("Amplitude")
            main_bar.update(1)

            ## Mel Spectrogram
            main_bar.pbar.set_description("Generating Mel Spectrogram..: n_mels: 128, fmax: 8000")
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=f_length, hop_length=h_length, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, hop_length=h_length, fmax=8000, ax=axes[1], cmap='magma')
            axes[1].set_title(f"{filename} - Mel Spectrogram", weight='bold')
            fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
            main_bar.update(1)

            ## Harmonic vs Percussive
            main_bar.pbar.set_description("Generating Harmonic vs Percussive..")
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            D_harmonic = librosa.amplitude_to_db(np.abs(librosa.stft(y_harmonic, n_fft=f_length, hop_length=h_length)), ref=np.max)
            librosa.display.specshow(D_harmonic, sr=sr, hop_length=h_length, x_axis='time', y_axis='log', ax=axes[2])
            axes[2].set_title(f"{filename} - Harmonic Content (Vocals/Melody)", weight='bold')
            main_bar.update(1)

            ## Pitch Tracking (F0)
            main_bar.pbar.set_description("Generating Pitch Tracking (F0)..: min note: `C2`, max note: `C7`, fmax: 8000")
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=f_length, hop_length=h_length)
            times = librosa.times_like(f0, sr=sr, hop_length=h_length)
            librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, hop_length=h_length, fmax=8000, ax=axes[3], cmap='magma')
            axes[3].plot(times, f0, label='f0', color='cyan', linewidth=2.5)
            axes[3].set_title(f"{filename} - Pitch Tracking (F0 Overlay)", weight='bold')
            axes[3].legend(loc='upper right')
            main_bar.update(1)

            ## Chromagram
            main_bar.pbar.set_description("Generating Chromagram..")
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=f_length, hop_length=h_length)
            img_chroma = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', hop_length=h_length, ax=axes[4], cmap='coolwarm')
            axes[4].set_title(f"{filename} - Chromagram (Musical Notes)", weight='bold')
            fig.colorbar(img_chroma, ax=axes[4], pad=0.01)
            main_bar.update(1)

            main_bar.pbar.set_description("Saving image result..")
            safe_title = "".join([c for c in filename if c.isalpha() or c.isdigit() or c in ' _-']).strip()
            search_pattern = str(Path(config["data_dir"]) / "*" / safe_title)
            matching_dirs = glob.glob(search_pattern)
            if matching_dirs:
                dir_path = Path(matching_dirs[0]).parent
            else:
                filepath = f"{safe_title}_visualization"
                dir_path = Path(config["data_dir"]) / filepath
                dir_path.mkdir(parents=True, exist_ok=True)
            output_path = dir_path / f"{filename}_visualization.jpg"
            plt.tight_layout(h_pad=1.5)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Visualization saved to {output_path}")
            main_bar.update(1)
