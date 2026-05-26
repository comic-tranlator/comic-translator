from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from src.util.path import BASE_DIR


def get_cache_path_by_url(url) -> Path:
    parts = urlparse(url)

    hub_dir = BASE_DIR / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    return hub_dir / Path(parts.path).name


def download_model(url):
    cached_file = get_cache_path_by_url(url)
    if not cached_file.exists():
        print(f"Downloading {url} to {cached_file.relative_to(Path.cwd())}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        tmp_file = cached_file.with_suffix(".tmp")

        try:
            with tqdm(total=total_size, unit="B") as progress_bar:
                with open(tmp_file, "wb") as file:
                    for data in response.iter_content(1024):
                        progress_bar.update(len(data))
                        file.write(data)

            tmp_file.rename(cached_file)
        except:
            tmp_file.unlink(missing_ok=True)
            raise
    return cached_file
