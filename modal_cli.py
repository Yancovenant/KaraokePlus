# modal-cli
import modal

app = modal.App("Kplus Testing Env")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("ffmpeg")
    .run_commands("pip install torch setuptools wheel numpy librosa matplotlib qwen_asr torchvision torchaudio tqdm yt-dlp stable-ts faster-whisper")
    .uv_pip_install("scipy", "sequence_align", "diffq", "demucs")
    .add_local_python_source("kplus")
    .add_local_dir("../Vocs", remote_path="/root/Vocs")
)
@app.function(image=image, gpu="T4")
def run_kplus_remote(*arglist):
    import kplus.cli
    import sys
    sys.argv = ["kplus"] + list(arglist)
    kplus.cli.main()


if __name__ == "__main__":
    print("Starting Modal automation run...")
    with app.run():
        run_kplus_remote.remote()
    print("Modal run finished successfully.")