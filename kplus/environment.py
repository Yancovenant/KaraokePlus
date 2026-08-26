from __future__ import annotations

import gc
import hashlib
import importlib
import logging
import os
import platform
import random
import shutil
import signal
import socket
import subprocess
import collections
import sys
import uuid
import warnings
from functools import cached_property, wraps
from pathlib import Path

import kplus

#from kplus.tools.progress import MainProgress


logger = logging.getLogger(__name__)

try:
    # available since python 3.13
    from warnings import deprecated
except ImportError:
    # simplified version
    class deprecated:
        def __init__(
            self,
            message: str,
            /,
            *,
            category: type[Warning] | None = DeprecationWarning,
            stacklevel: int = 1,
        ) -> None:
            if not isinstance(message, str):
                raise TypeError(
                    f"Expected an object of type str for 'message', not {message.__class__.__name__!r}",
                )
            self.message = message
            self.category = category
            self.stacklevel = stacklevel

        def __call__(self, obj, /):
            message = self.message
            category = self.category
            stacklevel = self.stacklevel
            if category is None:
                obj.__deprecated__ = message
                return obj
            if callable(obj):
                @wraps(obj)
                def wrapper(*args, **kwargs):
                    warnings.warn(message, category=category, stacklevel=stacklevel + 1)
                    return obj(*args, **kwargs)

                obj.__deprecated__ = wrapper.__deprecated__ = message
                return wrapper
            raise TypeError(f"@deprecated decorator cannot be applied to {obj!r}")


class environment:
    def __init__(self):
        self.is_colab = "COLAB_RELEASE_TAG" in os.environ or Path("./content").exists()
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("./kaggle").exists()
        self.is_docker = Path("./.dockerenv").exists()
        self.is_local = not any([self.is_colab, self.is_kaggle, self.is_docker])
        self.print_banner()
        
    def print_banner(self):
        # Use module level rich first, since tools.rich is overlapping with env
        self.rich  # noqa: B018
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.table import Column, Table
        from rich.text import Text

        from .ansii_logo import all_logos
        
        console = Console()
        def apply_color_layers(raw_logo: str) -> str:
            """Applies a dynamic top-down gradient to ASCII text based on its height."""
            themes = [
                ["cyan", "bright_cyan", "blue", "magenta"],                  # Synthwave
                ["bright_yellow", "yellow", "orange3", "dark_orange"],       # Sunset
                ["bright_green", "green", "spring_green2", "sea_green2"],    # Neon Matrix
                ["bright_magenta", "magenta", "purple", "deep_pink2"]        # Vaporwave
            ]
            palette = random.choice(themes)
            # Remove leading/trailing empty newlines so the gradient calculates correctly
            lines = raw_logo.strip("\n").split("\n")
            colored_lines = []
            for i, line in enumerate(lines):
                # Calculate which color layer this row belongs to
                color_index = int((i / len(lines)) * len(palette))
                if color_index >= len(palette):
                    color_index = len(palette) - 1
                color = palette[color_index]
                colored_lines.append(f"[{color}]{line}[/]")
            return "\n".join(colored_lines)

        logo = random.choice(all_logos)
        logo = apply_color_layers(logo)
        info_table = Table.grid(Column(), Column(ratio=1), expand=True, padding=(0, 0))
        info_table.add_row(Text("Author", style="b color(15)"), Text(": " + kplus.Release.author))
        info_table.add_row(Text("Version", style="b color(15)"), Text(": " + kplus.Release.version))
        for k, v in self.sys_info.items():
            info_table.add_row(Text(f"{k}", style="b color(15)"), Text(": " + v, no_wrap=True))
        main_panel = Panel(
            Group(logo, Text(""), info_table),
            border_style="color(14)",
            title=Text("Karaoke+", style="b"),
            subtitle=Text("Ready", style="dim"),
            expand=True, padding=1
        )
        console.print(main_panel)
        console.print()
    
    @cached_property
    def sys_info(self) -> dict:
        env_mapping = {
            "colab": self.is_colab,
            "kaggle": self.is_kaggle,
            "docker": self.is_docker,
            "local": self.is_local
        }
        environment = next((env for env, active in env_mapping.items() if active), "unknown")
        host = socket.gethostname()
        mac_node = str(uuid.getnode())
        unique_string = f"{host}-{mac_node}".encode()
        short_hash = hashlib.sha256(unique_string).hexdigest()[:12].upper()
        return {
            "Session": f"{host}-{short_hash}",
            "Platform": platform.platform(),
            "System": platform.system().lower(),
            "OS": os.name,
            "Environment": environment,
            "Python": sys.version.split()[0],
        }

    def run_cmd(self, cmd: list[str]) -> None:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
     
    def _run_cmd_install(self, name: str):
        return self.run_cmd([sys.executable, "-m", "pip", "install", "-q", name])
        
    def _get_pkg(self, name: str):
        if name == "stable_ts":
            install_name, import_name = "stable-ts", "stable_whisper"
        elif name == "onnxruntime_gpu":
            install_name, import_name = "onnxruntime-gpu==1.26.0", "onnxruntime"
        else:
            install_name, import_name = name, name
        def attempt_import():
            return importlib.import_module(import_name)
        def attempt_install_and_import():
            if install_name == "tqdm" or not env:
                self._run_cmd_install(install_name)
            else:
                cmd = [sys.executable, "-m", "pip", "install", install_name, "--progress-bar", "off"]
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, # This ensures we capture sys log output.
                    text=True, bufsize=1
                )
                from kplus.tools import rich
                max_lines = 6
                output_queue = collections.deque(maxlen=max_lines)
                lines, last_line = [], None
                with rich.Live(transient=True, refresh_per_second=15) as live:
                    for line in iter(process.stdout.readline, ''):
                        if not (line:=line.strip()): continue
                        if line == last_line: continue
                        if len(lines) >= max_lines: lines.pop()
                        lines.append(line)
                        install_text = rich.Text("\n".join(lines))
                        live.update(rich.Panel(install_text, title=f"Installing {install_name}...", style="color(14)", padding=1))
                process.stdout.close()
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        f"Subprocess failed with exit code {return_code}.\n"
                        f"Command: {' '.join(cmd)}\n")
            return importlib.import_module(import_name)
        for fn in [attempt_import, attempt_install_and_import]:
            try:
                return fn()
            except Exception as err:  # noqa: BLE001
                logger.warning(f"Attempt failed for {name}: {err}")
                continue
        raise ImportError(f"!!! Cannot continue as {name} could not be installed or imported...")
    
    def _get_apt(self, binary_name:str):
        if (exist:=shutil.which(binary_name)): return exist
        try:
            if os.name == "nt":
                pkg_map = {
                    "ffmpeg": "Gyan.FFmpeg",
                    "nodejs": "OpenJS.NodeJS",
                    "deno": "DenoLand.Deno"
                }
                pkg_name = pkg_map.get(binary_name, binary_name)
                self.run_cmd(["winget", "install", "--accept-source-agreements", "--accept-package-agreements", "--no-upgrade", pkg_name])
            else:
                if binary_name == "deno":
                    subprocess.run("curl -fsSL https://deno.land/install.sh | sh", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    deno_home = Path.home() / ".deno" / "bin"
                    os.environ["PATH"] = f"{deno_home}{os.pathsep}{os.environ['PATH']}"
                else:
                    self.run_cmd(["sudo", "apt-get", "update", "-y", "-qq"])
                    self.run_cmd(["sudo", "apt-get", 'install', "-y", "-qq", binary_name])
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
        self.torch  # noqa: B018
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def _signal_handler(self, sig, frame):
        print("TODO: Process is stopping by signal", signal.Signals(sig).name, frame)
        if sig in [signal.SIGINT, signal.SIGTERM]:  # noqa: SIM114
            pass
        elif hasattr(signal, 'SIGXCPU') and sig == signal.SIGXCPU:  # noqa: SIM114
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
        self.rich  # noqa: B018
        self._setup_signal()
        from rich.traceback import install
        install()

    @property
    def verbose(self) -> int:
        #TODO connect this with config later on
        return 1

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
        "torchcrepe", "scipy", "qwen_asr", "rich", "torchvision", "plotly",
        "plotext", "audio-separator", "plotly_resampler", "pypinyin", "anyascii", "jellyfish"]
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
