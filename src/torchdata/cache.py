import os
import shutil
from pathlib import Path


_cache = None


def set_cache_directory(cache_dir):
    global _cache
    if not cache_dir:
        _cache = DummyCache()
    else:
        _cache = FsCache(cache_dir)


class DummyCache:
    def __contains__(self, item):
        return False

    def add_file(self, item, file_path):
        pass


class FsCache:
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir).absolute()

    def __contains__(self, item):
        if item is None:
            return False
        return (self.cache_dir / item).is_file()

    def add_file(self, item, source_path):
        shutil.copy(str(source_path), str(self.cache_dir / item))

    def get_file(self, item, dest_path=None):
        source_path = self.cache_dir / item
        if dest_path is None:
            return source_path
        else:
            shutil.copy(str(source_path), str(dest_path))
            return Path(dest_path)


def get_cache():
    return _cache


set_cache_directory(os.environ.get('TORCHDATA_CACHE', None))
