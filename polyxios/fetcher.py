import os
from os.path import expanduser, join as pjoin

import requests
from tqdm import tqdm

from polyxios.exceptions import FetcherError

POLYXIOS_HOME = os.getenv("POLYXIOS_HOME", pjoin(expanduser("~"), ".polyxios"))

DATA_BASE_URL = (
    "https://raw.githubusercontent.com/ganimtron-10/polyxios-data/initial-models/"
)


def fetch(filename: str, overwrite: bool = False) -> str:
    """
    Resolve, download, and track local path for any Polyxios test asset.

    Parameters
    ----------
    filename : str
        The name of the file to fetch (e.g., 'stanford-bunny.obj').
    overwrite : bool, optional
        Force re-download of the asset even if it exists locally.

    Returns
    -------
    str
        The absolute local path to the fetched file.
    """
    filename_lower = filename.lower()

    _, ext = os.path.splitext(filename_lower)
    if not ext:
        raise FetcherError(
            f"Cannot resolve target folder: filename '{filename}' has no extension."
        )
    subfolder = ext[1:]

    target_dir = pjoin(POLYXIOS_HOME, subfolder)
    target_path = pjoin(target_dir, filename)

    if os.path.exists(target_path) and not overwrite:
        return target_path

    os.makedirs(target_dir, exist_ok=True)
    remote_url = f"{DATA_BASE_URL}{subfolder}/{filename}"

    try:
        with requests.get(remote_url, stream=True, timeout=15) as response:
            if response.status_code == 404:
                raise FetcherError(
                    f"Asset '{filename}' not found at remote path: {remote_url}"
                )
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with (
                open(target_path, "wb") as f,
                tqdm(
                    desc=f"Fetching {filename}",
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    leave=False,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    except requests.RequestException as e:
        if os.path.exists(target_path):
            os.remove(target_path)
        raise FetcherError(f"Failed to synchronize asset '{filename}': {e}") from e

    return target_path


def fetch_by_extension(ext: str, overwrite: bool = False) -> list[str]:
    """
    Discover and download all remote assets matching a specific file extension.

    Parameters
    ----------
    ext : str
        The extension to query (e.g., '.obj' or 'obj').
    overwrite : bool, optional
        Force re-download of all discovered assets.

    Returns
    -------
    list of str
        The absolute local paths to all fetched files.
    """
    ext_clean = ext.lower().lstrip(".")
    if not ext_clean:
        raise FetcherError("Invalid extension format provided.")

    api_url = f"https://api.github.com/repos/ganimtron-10/polyxios-data/contents/{ext_clean}?ref=initial-models"
    headers = {"User-Agent": "polyxios-fetcher"}

    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        response = requests.get(api_url, headers=headers, timeout=15)
        if response.status_code == 403 and "rate limit" in response.text.lower():
            raise FetcherError(
                "GitHub API rate limit reached. Set the 'GITHUB_TOKEN' environment "
                "variable to bypass rate-limiting blocks."
            )
        response.raise_for_status()
        directory_contents = response.json()
    except requests.RequestException as e:
        raise FetcherError(
            f"Failed to query remote repository directory for extension '{ext_clean}': {e}"
        ) from e

    discovered_files = [
        item["name"]
        for item in directory_contents
        if item["type"] == "file" and item["name"].lower().endswith(f".{ext_clean}")
    ]

    if not discovered_files:
        print(f"No files discovered matching extension '.{ext_clean}'.")
        return []

    print(f"Found {len(discovered_files)} assets for '.{ext_clean}'. Synchronizing...")
    local_paths = [
        fetch(filename, overwrite=overwrite) for filename in discovered_files
    ]

    return local_paths
