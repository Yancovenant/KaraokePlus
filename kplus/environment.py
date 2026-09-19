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
import sys
import uuid
import warnings
from functools import cached_property, partial, wraps
from pathlib import Path

import kplus
from kplus.ansii_logo import all_logos

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


class EnvironmentManager:
    def __init__(self):
        self._detect_environment()
        if self.is_colab:
            from google.colab import output
            output.enable_custom_widget_manager()
            logger.debug("Google colab enabled custom widget manager")

    def _detect_environment(self) -> None:
        self.is_colab = "COLAB_RELEASE_TAG" in os.environ or Path("./content").exists()
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("./kaggle").exists()
        self.is_docker = Path("./.dockerenv").exists()
        self.is_local = not any([self.is_colab, self.is_kaggle, self.is_docker])

    # Helper
    def clean(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        gc.collect()

    # Property
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
    
    @property
    def verbose(self) -> int:
        #TODO connect this with config later on
        return 1

    @cached_property
    def device(self):
        return self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")

    @cached_property
    def device_count(self):
        return self.torch.cuda.device_count()

    @cached_property
    def fonts(self):
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

    
    # Lazy Property Module
    def resolve_pkg(self, pkg_name: str, version: str):
        import_name = pkg_name
        install_name = pkg_name
        if pkg_name == "stable-ts":
            import_name = "stable_whisper"
        
        def _try_import():
            return importlib.import_module(import_name)
        
        def _try_install():
            name = install_name.replace("_", "-")
            name_version = f"{name}{version}"
            cmd = [
                sys.executable, "-m", "pip",
                "install", name_version,
                "--progress-bar", "off",
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, # This ensures we capture sys log output.
                text=True, bufsize=1
            )
            def _finish():
                process.stdout.close()
                return_code = process.wait()
                if return_code != 0:
                    raise RuntimeError(
                        f"Download package failed `{name_version}`. return_code: {return_code}.\n"
                        f"Command: {' '.join(cmd)}\n"
                    )
                return _try_import()
            # Require Rich
            if pkg_name == "rich":
                return _finish()
            from kplus.tools import rich
            max_lines = 6
            lines, last_line = [], None
            with rich.Live(transient=True, refresh_per_second=15) as live:
                for line in iter(process.stdout.readline, ''):
                    if not (line:=line.strip()):
                        continue
                    if line == last_line:
                        continue
                    if len(lines) >= max_lines:
                        lines.pop(0)
                    install_text = rich.Text("\n".join(lines))
                    live.update(
                        rich.Panel(
                            install_text,
                            title=f"Installing {install_name}...",
                            style="color(14)",
                            padding=1
                        )
                    )
            return _finish()
        for fn in (_try_import, _try_install):
            try:
                return fn()
            except ImportError:
                logger.debug(f"Import error for `{import_name}` trying to install...")
                continue
            except Exception as err:  # noqa: BLE001
                logger.warning(f"Download Attempt `{fn.__name__} failed for `{install_name}`: {err}")
        raise ImportError(f"Connot continue as {import_name} could not be installed or imported.")

    def resolve_apt(self, apt_name: str, windows_name: str):
        if (exist:=shutil.which(apt_name)):
            return exist
        try:
            if os.name == "nt":
                cmds = [f"winget install --accept-source-agreements --accept-package-agreements --no-upgrade {windows_name}"]
            else:
                if apt_name == "deno":
                    # Deno is special
                    cmds = ["curl -fsSL https://deno.land/install.sh | sh"]
                else:
                    cmds = [
                        "sudo apt-get update -y -qq",
                        f"sudo apt-get install -y -qq {apt_name}",
                    ]
            for cmd in cmds:
                subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if apt_name == "deno":
                deno_home = Path.home() / ".deno" / "bin"
                os.environ["PATH"] = f"{deno_home}{os.pathsep}{os.environ['PATH']}"
        except subprocess.CalledProcessError as e:
            logger.warning(f"Package manager failed to install {apt_name}: {e}")
        if not (exist:=shutil.which(apt_name)):
            raise RuntimeError(f"Connot continue as {apt_name} could not be found in PATH after installation attempt.")
        return exist
    
    # Setup
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

    def print_banner(self):
        from kplus.tools import rich

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
        info_table = rich.Table.grid(
            rich.Column(),
            rich.Column(ratio=1),
            expand=True,
            padding=(0, 0)
        )
        info_table.add_row(
            rich.Text("Author", style="b color(15)"),
            rich.Text(": " + kplus.Release.author)
        )
        info_table.add_row(
            rich.Text("Version", style="b color(15)"),
            rich.Text(": " + kplus.Release.version)
        )
        for k, v in self.sys_info.items():
            info_table.add_row(
                rich.Text(f"{k}", style="b color(15)"),
                rich.Text(": " + v, no_wrap=True)
            )
        main_panel = rich.Panel(
            rich.Group(logo, rich.Text(""), info_table),
            border_style="color(14)",
            title=rich.Text("Karaoke+", style="b"),
            subtitle=rich.Text("Ready", style="dim"),
            expand=True, padding=1
        )
        rich.print(main_panel)
        rich.print()

    def setup_environment(self):
        self._setup_signal()

        self.rich  # noqa: B018
        from rich.traceback import install
        install()

        # Tqdm with rich error
        self.tqdm  # noqa: B018
        import tqdm
        tqdm.tqdm = partial(tqdm.tqdm, file=sys.stdout, disable=True)

        self.print_banner()


REQUIRED_PKG: dict[str, str] = {
    # Song Download Requirements
    "yt-dlp": "",
    "yt-dlp-ejs": "",
    # Torch
    "torch": "",
    "torchaudio": "",
    "torchvision": "",
    # Audio Separation Requirements
    "demucs": "",
    "diffq": "",
    # Audio Detection
    "librosa": "",
    "scipy": "",
    "matplotlib": "",
    "plotly": "",
    "plotext": "",
    "plotly_resampler": "",
    # ASR Requirements
    "transformers": ">=5.17.0",
    "numpy": "<=2.2",
    "stable-ts": "",
    "faster-whisper": "",
    "langdetect": "",
    # Lyric Alignment Requirements
    "sequence_align": "",
    # Text Normalization
    "pypinyin": "",
    "pykakasi": "",
    "anyascii": "",
    "jellyfish": "",
    # Misc
    "rich": "",
    "requests": "",
    "tqdm": "", # For error combining with rich
    # Worker
    "kaggle": "",
}

def wrap_pkg(pkg_name: str, version: str):
    def getter(self):
        return self.resolve_pkg(pkg_name, version)
    getter.__name__ = pkg_name
    cp = cached_property(getter)
    cp.__set_name__(EnvironmentManager, pkg_name)
    return cp

for pkg_name, version in REQUIRED_PKG.items():
    pkg_name = pkg_name.replace("-", "_")
    setattr(EnvironmentManager, pkg_name, wrap_pkg(pkg_name, version))


REQUIRED_PKG_OLD = [
        "", "soundfile", "", "", "",
        "rapidfuzz", "", "zhconv", "", "korean_romanizer",
        "google-api-python-client", "google-auth-httplib2", "google-auth-oauthlib",
        "", "", "", "", "", "tqdm", "",
        "psutil", "", "", "", "onnxruntime", "onnxruntime-gpu",
        "torchcrepe", "", "qwen_asr", "", "", "",
        "", "audio-separator", "", "", "", "",
        ""]

REQUIRED_APT: dict[str, str] = {
    "ffmpeg": "Gyan.FFmpeg",
    "deno": "DenoLand.Deno",
    "nodejs": "OpenJS.NodeJS",
}

def wrap_apt(apt_name: str, windows_name: str):
    def getter(self):
        return self.resolve_apt(apt_name, windows_name)
    getter.__name__ = apt_name
    cp = cached_property(getter)
    cp.__set_name__(EnvironmentManager, apt_name)
    return cp

for apt_name, windows_name in REQUIRED_APT.items():
    setattr(EnvironmentManager, apt_name, wrap_apt(apt_name, windows_name))

env = EnvironmentManager()

if __name__ == "__main__":
    print("======= Test Environment =======")
    print("1. Test Setup Environment")
    env.setup_environment()
    print("2. Test Install Everything")
    env.rich.inspect(env)

