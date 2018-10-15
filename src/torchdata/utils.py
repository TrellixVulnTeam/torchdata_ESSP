import hashlib
import tempfile
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm

from torchdata.cache import get_cache


def _create_urllib_reporthook(progress_bar):
    """Create a urllib reporthook for updating a progress bar.

    Args:
        progress_bar (tqdm): A progress bar instance.

    Returns:
        A urllib progress callback function.
    """
    def update_progress(count, block_size, total_size):
        progress_bar.total = total_size
        n_bytes_received = count * block_size
        progress_bar.update(n_bytes_received - progress_bar.n)
    return update_progress


@contextmanager
def temporary_directory(suffix='', prefix='tmp'):
    """Create a temporary directory.

    Returns:
        Path: Path to the temporary directory.
    """
    with tempfile.TemporaryDirectory(suffix, prefix) as temp_dir:
        yield Path(temp_dir)


def md5sum(file_path, quiet=False):
    file_path = Path(file_path)
    md5_algorithm = hashlib.md5()
    with file_path.open('rb') as f:
        stat = file_path.stat()
        if quiet:
            progress_bar = None
        else:
            progress_bar = tqdm(desc='Calculating checksum', unit='B', unit_scale=True, ascii=True,
                                leave=False, total=stat.st_size)
        for chunk in iter(lambda: f.read(stat.st_blksize), b''):
            md5_algorithm.update(chunk)
            if progress_bar:
                progress_bar.update(len(chunk))
        if progress_bar:
            progress_bar.close()
    return md5_algorithm.hexdigest()


def download_file(url, dest_path, md5=None, quiet=False):
    """Download a remote file to a specific location."""
    cache = get_cache()
    # If the file is already in the cache, copy it
    if md5 in cache:
        cache.get_file(md5, dest_path)
        return
    # Otherwise download the file
    if quiet:
        progress_bar = None
        reporthook = None
    else:
        progress_bar = tqdm(desc='Downloading', unit='B', unit_scale=True, ascii=True,
                            leave=False)
        reporthook = _create_urllib_reporthook(progress_bar)
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    if progress_bar:
        progress_bar.close()
    actual_md5 = md5sum(dest_path, quiet=quiet)
    if md5:
        assert actual_md5 == md5, \
               'file integrity check failed (expected {}, got {})'.format(md5, actual_md5)
    cache.add_file(actual_md5, dest_path)


@contextmanager
def remote_file(url, md5=None, quiet=False):
    """Get a local filesystem reference to a remote file."""
    cache = get_cache()
    if md5 in cache:
        # Yield the cached file directly
        yield cache.get_file(md5)
    else:
        # Download the file to a temporary location
        with temporary_directory('torchdata') as temp_dir:
            dl_file = temp_dir / 'download'
            download_file(url, dl_file, md5, quiet)
            yield dl_file
