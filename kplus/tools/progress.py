from pathlib import Path

import kplus


class SubProgress:
    def __init__(self, leave: bool = False):
        # self.leave = leave
        self.pbar = None
    def __call__(self, d: dict):
        from tqdm import tqdm
        status = d.get('status')
        
        if status == 'downloading':
            total = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            downloaded = d.get('downloaded_bytes', 0)
            
            # Spawn the bar if it doesn't exist and we know the file size
            if self.pbar is None and total > 0:
                filename = Path(d.get('filename', 'Unknown File')).name
                # Shorten long filenames so they don't break the terminal UI
                desc = f"  ↳ {filename[:25]}..." if len(filename) > 25 else f"  ↳ {filename}"
                self.pbar = tqdm(
                    total=total, 
                    unit='iB', 
                    unit_scale=True, 
                    unit_divisor=1024,
                    # position=self.position, 
                    desc=desc, 
                    # leave=False # Deletes the bar when finished!
                )
            
            if self.pbar:
                # Update relative to what we've already downloaded
                self.pbar.update(downloaded - self.pbar.n)
                
        elif status == 'finished':
            if self.pbar:
                self.pbar.close()
                self.pbar = None

class MainProgress:
    def __init__(self, total: int,
            desc: str = "Total Progress",
            position: int = 0, unit: str = "task", **kwargs):
        kplus.env.tqdm
        from tqdm import tqdm
        if desc:
            tqdm.write(f"\n{desc}")
        format = "{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total} [{elapsed}<{remaining}] {postfix}"
        self.pbar = tqdm(total=total, desc=" ↳ Status :",
                position=position, leave=True, bar_format=format,
                unit=unit, **kwargs)

    def update(self, n: int = 1):
        self.pbar.update(n)

    def __enter__(self):
        from tqdm.contrib.logging import logging_redirect_tqdm
        self._log_ctx = logging_redirect_tqdm()
        self._log_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar is not None:
            self.pbar.close()
        if self._log_ctx is not None:
            self._log_ctx.__exit__(exc_type, exc_val, exc_tb)
            
