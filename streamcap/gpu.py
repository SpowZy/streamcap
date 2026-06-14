"""CUDA runtime discovery for ctranslate2 on Windows.

The pip wheels `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` ship the DLLs
ctranslate2 needs for GPU inference, but Windows won't find them unless the
package `bin` directories are registered before the model loads.
"""

from __future__ import annotations

import importlib.util
import os


def prepare_cuda_dlls() -> None:
    for pkg in ("nvidia.cublas", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            bin_dir = os.path.join(list(spec.submodule_search_locations)[0], "bin")
            if os.path.isdir(bin_dir):
                os.add_dll_directory(bin_dir)
