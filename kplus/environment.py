import sys
import kplus
import socket
import uuid
import hashlib
import platform
import os
import logging
import subprocess
import gc
import importlib
import signal
import shutil

from functools import cached_property
from pathlib import Path

from kplus.tools.progress import MainProgress


logger = logging.getLogger(__name__)


class environment:
    def __init__(self):
        self.is_colab = "COLAB_RELEASE_TAG" in os.environ or Path("./content").exists()
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("./kaggle").exists()
        self.is_docker = Path("./.dockerenv").exists()
        self.is_local = not any([self.is_colab, self.is_kaggle, self.is_docker])
        self.print_banner()
        
    def print_banner(self):
        info_lines = [f"{k:<11}: {v}" for k, v in self.sys_info.items()]
        max_info_len = max(len(line) + 11 for line in info_lines)
        max_len = max(max_info_len, len(kplus.Release.description) + 10, 40)
        banner_parts = ['*' * max_len,
                        kplus.Release.description.center(max_len, " "),
                        f" version: {kplus.Release.version} ".center(max_len, " "),
                        " INFORMATION ".center(max_len, "-"),
                        *info_lines,
                        '*' * max_len]
        print("\n".join(banner_parts))
    
    @cached_property
    def sys_info(self) -> dict:
        env_mapping = {"colab": self.is_colab,
                                     "kaggle": self.is_kaggle,
                                     "docker": self.is_docker,
                                     "local": self.is_local}
        environment = next((env for env, active in env_mapping.items() if active), "unknown")
        host = socket.gethostname()
        mac_node = str(uuid.getnode())
        unique_string = f"{host}-{mac_node}".encode("utf-8")
        short_hash = hashlib.sha256(unique_string).hexdigest()[:12].upper()
        return {"Session": f"{host}-{short_hash}",
                     "Platform": platform.platform(),
                     "OS": os.name,
                     "Environment": environment,
                     "Python": sys.version.split()[0],
                     "Torch": self.device.type}
                     
    @cached_property
    def uv(self) -> list[str]:
        try:
            self._run_sys_cmd(["uv", "--version"])
        except Exception:
            logger.warning(">> UV package is not installed, falling back using pip")
            return ["pip"]
        # return ["uv", "pip"] # UV still different in different platform
        return ["pip"]
     
    def _run_sys_cmd(seld, cmd: list[str]):
        return subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    def _run_py_cmd(self, cmd:list[str]):
        cmd = [sys.executable, "-m"] + cmd
        return self._run_sys_cmd(cmd)
    
    def _run_cmd_install(self, name: str):
        return self._run_py_cmd(self.uv + ["install", "-q", name])
        
    def _get_pkg(self, name: str):
        def attempt_import():
            return importlib.import_module(name)
        def attempt_install_and_import():
            if name == "tqdm" or not env:
                self._run_cmd_install(name)
            else:
                cmd = [sys.executable, "-m", *self.uv, "install", name, "--progress-bar", "off"]
                process = subprocess.Popen(cmd,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, # This ensures we capture sys log output.
                        text=True, bufsize=1)
                full_output_log = []
                with MainProgress(total=0, desc=f"Package '{name}' missing. Installing...", unit="pkg") as main_bar:
                    downloaded_count = 0
                    pbar = main_bar.pbar
                    for line in iter(process.stdout.readline, ''):
                        if not (clean_line:=line.strip()): continue
                        full_output_log.append(clean_line)
                        logger.debug(f">> {self.uv[0]}: {clean_line}")
                        if clean_line.startswith("Collecting "):
                            downloaded_count += 1
                            pbar.total = downloaded_count
                            pbar.set_description(f"Downloading {name} & deps")
                            pbar.refresh()
                        elif clean_line.startswith("Installing collected packages:"):
                            packages_str = clean_line.split(":", 1)[1].strip()
                            total_to_install = len([p for p in packages_str.split(",") if p.strip()])
                            pbar.total = total_to_install
                            pbar.n = 0
                            pbar.set_description(f"Installing {total_to_install} packages")
                            pbar.refresh()
                    if pbar.total > 0:
                        pbar.n = pbar.total
                        pbar.set_description(f"Finished {name}")
                        pbar.refresh()
                    main_bar.update(1)
                process.stdout.close()
                return_code = process.wait()
                if return_code != 0:
                    error_context = "\n".join(full_output_log[-15:])
                    logger.error("\n" + "="*50)
                    logger.error(f" PIP INSTALLATION FAILED FOR: {name}")
                    logger.error("="*50)
                    logger.error(error_context)
                    logger.error("="*50 + "\n")
                    raise RuntimeError(
                        f"Subprocess failed with exit code {return_code}.\n"
                        f"Command: {' '.join(cmd)}\n"
                        f"Last output:\n{error_context}")
            return importlib.import_module(name)
        for fn in [attempt_import, attempt_install_and_import]:
            try:
                return fn()
            except Exception as err:
                logger.warning(f"Attempt failed for {name}: {err}")
                continue
        raise ImportError(f"!!! Cannot continue as {name} could not be installed or imported...")
    
    def _get_apt(self, binary_name:str):
        if (exist:=shutil.which(binary_name)): return exist
        try:
            if os.name == "nt":
                pkg_map = {"ffmpeg": "Gyan.FFmpeg",
                                     "nodejs": "OpenJS.NodeJS",
                                     "deno": "DenoLand.Deno"}
                pkg_name = pkg_map.get(binary_name, binary_name)
                subprocess.run(
                        ["winget", "install", "--accept-source-agreements", "--accept-package-agreements", "--no-upgrade", pkg_name], 
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            else:
                if binary_name == "deno":
                    subprocess.run("curl -fsSL https://deno.land/install.sh | sh", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    deno_home = Path.home() / ".deno" / "bin"
                    os.environ["PATH"] = f"{deno_home}{os.pathsep}{os.environ['PATH']}"
                else:
                    self._run_sys_cmd(["sudo", "apt-get", "update", "-y", "-qq"])
                    self._run_sys_cmd(["sudo", "apt-get", 'install', "-y", "-qq", binary_name])
        except Exception as e:
            raise Exception(f"!!! Command failed while trying to install {binary_name}: {e}")
        if not (final_path := shutil.which(binary_name)):
            raise Exception(f"!!! Cannot continue. {binary_name} could not be found in PATH after installation attempt.")
        return final_path
    
    @cached_property
    def _ensure_fonts_installed(self) -> bool:
        # required font, fonts-noto-cjk, montserrat bold
        if os.name != "nt":
            try:
                font_cmds = [
                        "sudo apt-get update -y -qq",
                        "sudo apt-get install -y -qq fonts-noto-cjk",
                        "wget -q https://github.com/JulietaUla/Montserrat/archive/refs/tags/v7.222.zip -O /tmp/montserrat.zip",
                        "unzip -q -o /tmp/montserrat.zip -d /tmp/montserrat",
                        "mkdir -p /usr/share/fonts/truetype/montserrat",
                        "cp /tmp/montserrat/Montserrat-7.222/fonts/ttf/* /usr/share/fonts/truetype/montserrat/"
                ]
                for cmd in font_cmds:
                    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception as e:
                raise Exception(f"!!! Command failed while trying to install fonts: {e}")
        raise Exception("!!! Cannot continue. as necessary fonts is not installed...")
        
    @cached_property
    def device(self):
        return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    def clean(self):
        self.torch
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def _signal_handler(self, sig, frame):
        print("TODO: Process is stopping by signal", signal.Signals(sig).name, frame)
        if sig in [signal.SIGINT, signal.SIGTERM]:
            pass
        elif hasattr(signal, 'SIGXCPU') and sig == signal.SIGXCPU:
            pass
        elif sig == signal.SIGHUP:
            pass
        sys.exit()

    def _setup_signal(self):
        if os.name != "nt":
            signal.signal(signal.SIGHUP, self._signal_handler)
            signal.signal(signal.SIGXCPU, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_environment(self):
        self.tqdm
        self._setup_signal()
    

def make_cached_pkg_wrap_methods(attr, owner_class):
    def getter(self):
        return self._get_pkg(attr)
    getter.__name__ = attr
    cp = cached_property(getter)
    cp.__set_name__(owner_class, attr)
    return cp
REQUIRED_PKG = [
        "yt-dlp", "soundfile", "torch", "torchaudio", "faster-whisper",
        "rapidfuzz", "pypinyin", "zhconv", "pykakasi", "korean_romanizer",
        "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib",
        "demucs", "diffq", "sequence_align", "stable-ts", "rich", "tqdm", "requests",
        "psutil", "librosa", "matplotlib", "numpy", "onnxruntime", "onnxruntime-gpu",
        "torchcrepe", "scipy"]
for pkg in REQUIRED_PKG:
    setattr(environment, pkg.replace("-", "_"), make_cached_pkg_wrap_methods(pkg.replace("-", "_"), environment))


def make_cached_apt_wrap_methods(attr, owner_class):
    def getter(self):
        return self._get_apt(attr)
    getter.__name__ = attr
    cp = cached_property(getter)
    cp.__set_name__(owner_class, attr)
    return cp
REQUIRED_APT = ["ffmpeg", "deno", "nodejs"]
for apt in REQUIRED_APT:
    setattr(environment, apt, make_cached_apt_wrap_methods(apt, environment))

env = environment()
