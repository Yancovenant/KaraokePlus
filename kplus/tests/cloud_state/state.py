import os
import pickle
import re
import shutil
from pathlib import Path

import boto3
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Column, Table
from rich.text import Text
from rich.tree import Tree

console = Console()

class State:
    def __init__(self, *to_track):
        self.tracked_vars = [*to_track]

        self._name = "CTCRefineVars"

        self.is_colab = "COLAB_RELEASE_TAG" in os.environ or Path("./content").exists()
        self.is_kaggle = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("./kaggle").exists()
        self.working_dir: Path = Path(
            "/content" if self.is_colab else
            "/kaggle/working" if self.is_kaggle else
            None
        ).resolve() / "CTCBackupData"
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.load_cloudflare()

    def get_secret(self, key: str) -> str:
        if self.is_colab:
            from google.colab import userdata
            return userdata.get(key)
        elif self.is_kaggle:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(key)
        raise RuntimeError("Environment not supported")

    def load_cloudflare(self) -> None:
        try:
            self.CF_ACCOUNT_ID: str = self.get_secret("CF_ACCOUNT_ID")
            self.CF_ACCESS_KEY: str = self.get_secret("CF_ACCESS_KEY")
            self.CF_SECRET_KEY: str = self.get_secret("CF_SECRET_KEY")
            self.CF_BUCKET_NAME: str = self.get_secret("CF_BUCKET_NAME")
            self.s3_client = boto3.client(
                's3',
                endpoint_url=f"https://{self.CF_ACCOUNT_ID}.eu.r2.cloudflarestorage.com",
                aws_access_key_id=self.CF_ACCESS_KEY,
                aws_secret_access_key=self.CF_SECRET_KEY,
                region_name="auto" # R2 requires region to be 'auto' or 'us-east-1'
            )
        except Exception as err:
            console.print(f"⚠️ Cloudflare secrets missing or invalid. Sync will fail. ({err})", style="b red")

    def _progress(self) -> Progress:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        )

    @staticmethod
    def unique_key(key: str) -> str:
        match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11}).*', key)
        if match:
            return match.group(1)
        # Fallback: sanitize the full URL into a safe string
        return re.sub(r'[^a-zA-Z0-9_-]', '_', key)

    def print_header(self, action: str, ukey: str) -> None:
        console.print()
        console.print(Panel(
            Text.assemble(
                Text("Target ID:    ", style="b white"),
                Text(ukey + "\n", style="cyan"),
                Text("Storage:      ", style="b white"),
                Text(str(self.working_dir), style="dim")
            ),
            title=f"CTC Refine | {action.upper()}",
            title_align="left",
            box=box.ROUNDED,
            expand=False
        ))

    def save(self, key: str, scope: dict) -> None:  # noqa: B008
        """ Save Data """
        ukey = self.unique_key(key)
        self.print_header("save", ukey)
        backup_dir: Path = Path(self.working_dir / ukey / self._name)
        backup_dir.mkdir(parents=True, exist_ok=True)
        save_path: Path = Path(self.working_dir / ukey / self._name).with_suffix(".pkl")
        temp_path: Path = Path(self.working_dir / ukey / self._name).with_suffix(".tmp")
        backup_path: Path = Path(self.working_dir / ukey / self._name).with_suffix(".bak")
        try:
            to_save: dict = {}
            file_to_save: list = []
            # Verbose
            tree = Tree("Payload Analysis", style="b white")
            var_node = tree.add("Variables:", style="blue")
            file_node = tree.add("Files to Backup", style="magenta")
            for name in self.tracked_vars:
                if name not in scope:
                    var_node.add(f"[dim red]✖[/] [dim]{name} (Not Initialized)[/]")
                    continue
                to_save[name] = scope[name]
                var_node.add(f"[green]✔[/] {name}")
                if name.endswith("path"):
                    file_to_save.append(name)
                    file_node.add(f"[green]✔[/] {name} [dim]({scope[name]})[/]")
            console.print(tree)
            with self._progress() as prg:
                if file_to_save:
                    task = prg.add_task("Backing up files...", total=len(file_to_save))
                    for name in file_to_save:
                        srcpath = Path(to_save[name]).resolve()
                        targetpath = backup_dir / srcpath
                        targetpath.parent.mkdir(parents=True, exist_ok=True)
                        if srcpath.is_file(): shutil.copy(srcpath, targetpath)
                        else: prg.console.print(f"⚠️ Warning: File '{srcpath}' missing. Variable saved without physical backup.", style="yellow")
                        prg.advance(task)
                task = prg.add_task("Serializing memory...", total=None)
                with open(temp_path, "wb") as f:
                    pickle.dump(to_save, f)
                    f.flush()
                    os.fsync(f.fileno())
                prg.advance(task)
                task = prg.add_task("Rotating backups...", total=None)
                if save_path.is_file():
                    if backup_path.is_file(): backup_path.unlink()
                    save_path.rename(backup_path)
                temp_path.rename(save_path)
                prg.advance(task)
                paths = sorted(
                    backup_dir.iterdir(),
                    key=lambda path: (path.is_file(), path.name.lower())
                )
                task = prg.add_task("Uploading to R2...", total=len(paths))
                for path in paths:
                    s3_key = str(path.relative_to(self.working_dir)).replace("\\", "/")
                    self.s3_client.upload_file(str(path), self.CF_BUCKET_NAME, s3_key)
                    prg.advance(task)
            console.print("\n✔ State safely secured to Cloudflare R2.\n", style="b green")
        except Exception as err:
            if temp_path.is_file(): temp_path.unlink()
            console.print(f"\n✖ Error during save sequence: {err}", style="b red")
            raise

    def load(self, key: str, scope: dict) -> None:
        """ Load Data """
        ukey = self.unique_key(key)
        self.print_header("load", ukey)
        backup_dir: Path = Path(self.working_dir / ukey / self._name)
        backup_dir.mkdir(parents=True, exist_ok=True)
        save_path: Path = Path(self.working_dir / ukey / self._name).with_suffix(".pkl")
        backup_path: Path = Path(self.working_dir / ukey / self._name).with_suffix(".bak")
        try:
            with console.status("Syncing from Cloudflare R2...") as status:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.CF_BUCKET_NAME)
                download_count = 0
                for page in pages:
                    if "Contents" not in page: continue
                    for obj in page["Contents"]:
                        localpath = self.working_dir / obj["Key"]
                        localpath.parent.mkdir(parents=True, exist_ok=True)
                        self.s3_client.download_file(self.CF_BUCKET_NAME, obj["Key"], localpath)
                        download_count += 1
                if download_count == 0:
                    console.print(f"⚠️ No objects found in bucket '{self.CF_BUCKET_NAME}'.", style="b yellow")
                status.update("Locating and verifying payload...")
                current_loadpath = save_path
                if not current_loadpath.is_file() or current_loadpath.stat().st_size == 0:
                    if backup_path.is_file() and backup_path.stat().st_size > 0:
                        console.print("⚠️ Main .pkl missing or empty. Reverting to .bak fallback.", style="b yellow")
                        current_loadpath = backup_path
                    else: raise FileNotFoundError("Neither main nor backup .pkl files could be found or read.")
                status.update("Deserializing memory map...", style="b cyan")
                with open(current_loadpath, "rb") as f:
                    saved_data = pickle.load(f)
                status.update("Injecting globals...", style="b cyan")
                for name, val in saved_data.items():
                    scope[name] = val
            file_to_copy: list[Path] = [
                name for name in self.tracked_vars
                if name.endswith("path")
                and name in saved_data
            ]
            if file_to_copy:
                with self._progress() as prg:
                    task = prg.add_task("Restoring physical files...", total=len(file_to_copy))
                    for name in file_to_copy:
                        if not scope[name]:
                            prg.advance(); continue
                        targetpath = Path(scope[name]).resolve()
                        srcpath = backup_dir / targetpath.relative_to("/")
                        if srcpath.is_file():
                            targetpath.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(srcpath, targetpath)
                        else:
                            prg.console.print(f"⚠️ Backup file missing: '{srcpath}'. Variable loaded without physical file.", style="yellow")
                        prg.advance(task)
            table = Table("Restored Environment",
                Column("Variable", style="b white"), Column("Object Type", style="cyan"),
                Column("Status", style="right"),
                box=box.SIMPLE_HEAVY, show_lines=True, expand=True
            )
            for name in self.tracked_vars:
                if name in saved_data:
                    table.add_row(name, type(saved_data[name]).__name__, "[green]✔ Restored[/]")
                else: table.add_row(name, "[dim]-[/]", "[dim red]✖ Missing[/]")
            console.print(table)
            console.print("\n✔ Global environment successfully restored.\n", style="b green")
        except Exception as e:
            console.print(f"\n[bold red]✖ Error during load sequence: {e}[/]")
            raise