from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


def _build_ext():
    # Import torch lazily so `pip` can at least import setup.py to show metadata.
    # Build should be run with `--no-build-isolation` so torch is available.
    from torch.utils.cpp_extension import BuildExtension, CppExtension

    root = Path(__file__).resolve().parent
    embedded = root / "sdpa-metal" / "embedded_metallib.h"
    if not embedded.exists():
        raise RuntimeError(f"Missing embedded metallib header: {embedded}")

    # IMPORTANT: setuptools requires *relative* paths for sources.
    sources = [
        "torch-ext/wheel_binding.cpp",
        "sdpa-metal/scaled_dot_product_attention.mm",
    ]

    extra_compile_args = {
        "cxx": [
            "-std=c++17",
            f'-DEMBEDDED_METALLIB_HEADER="{embedded}"',
            "-DEMBEDDED_METALLIB_NAMESPACE=metal_flash_sdpa_lib",
        ]
    }
    extra_link_args = [
        "-framework",
        "Metal",
        "-framework",
        "Foundation",
        "-framework",
        "MetalPerformanceShaders",
    ]

    ext_modules = [
        CppExtension(
            name="metal_flash_sdpa._C",
            sources=sources,
            include_dirs=["sdpa-metal", "torch-ext"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]

    cmdclass = {"build_ext": BuildExtension}
    return ext_modules, cmdclass


ext_modules, cmdclass = _build_ext()

setup(
    name="metal-flash-sdpa",
    version="0.1.2",
    description="Optimized SDPA kernels inspired by Flash Attention for Metal (MPS).",
    packages=find_packages(exclude=("tests", "build", "sdpa-metal", "torch-ext")),
    package_data={},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
    # NOTE: This extension requires PyTorch at build/runtime, but we intentionally do NOT
    # declare a strict dependency here to avoid pip "helpfully" upgrading/downgrading
    # the user's existing torch stack (common in conda envs).
    install_requires=[],
)

