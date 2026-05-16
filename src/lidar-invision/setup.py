"""
setup.py for lidar_invision — Python bindings for YDLIDAR rolling-buffer manager.

Structure mirrors cpp_inference/setup.py: platform detection → Pybind11Extension
→ post-install validation.

Build & install:
    pip install .          # production
    pip install -e .       # editable / dev
"""

import os
import platform
import subprocess
import sys

from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools.command.develop import develop
from setuptools.command.install import install


# ── Post-install validation ──────────────────────────────────────────────────

def run_validation():
    print("\n--- Running lidar_invision post-install validation ---")
    try:
        print(f"Python {sys.version_info.major}.{sys.version_info.minor} detected...")

        cmd = [
            sys.executable, "-c",
            (
                "from lidar_invision import LidarManager, Config; "
                "m = LidarManager(); "
                "print('Check 1/2: Module import OK'); "
                "print(f'Check 2/2: LidarManager instantiation OK — {m}')"
            ),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
        subprocess.check_call(cmd, env=env)
        print("RESULT: lidar_invision is READY.\n")

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        print("Tip: ensure YDLIDAR SDK is installed (see README).")
        print("     Run: ldd lidar_invision*.so  to check for missing symbols.\n")


class PostInstallCommand(install):
    def run(self):
        install.run(self)
        run_validation()


class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        run_validation()


# ── Platform detection ───────────────────────────────────────────────────────

system  = platform.system()   # 'Linux', 'Darwin', 'Windows'
machine = platform.machine()  # 'x86_64', 'aarch64', 'arm64'

is_pi  = system == "Linux" and machine.startswith(("arm", "aarch64"))
is_mac = system == "Darwin"

# ── Base build config ────────────────────────────────────────────────────────

include_dirs = [
    "src",                          # lidar_manager.hpp + bundled CYdLidar.h
    "/usr/local/include/ydlidar",   # SDK transitive headers (ydlidar_def.h etc.)
    "/usr/include/ydlidar",
]
library_dirs        = ["/usr/local/lib", "/usr/lib"]
libraries           = ["ydlidar_invision", "pthread"]
extra_compile_args  = ["-std=c++17", "-O3", "-Wall"]
extra_link_args     = ["-Wl,-rpath,/usr/local/lib"]

# ── Platform overrides ────────────────────────────────────────────────────────

if is_pi:
    # Raspberry Pi 4 / 5  (armv8.2)
    extra_compile_args += ["-march=armv8.2-a+dotprod"]
    print("[setup.py] Target: Raspberry Pi (aarch64)")

elif is_mac:
    # macOS / Apple Silicon — YDLIDAR SDK via homebrew or manual install
    include_dirs   += ["/opt/homebrew/include", "/opt/homebrew/include/ydlidar"]
    library_dirs   += ["/opt/homebrew/lib"]
    extra_link_args += ["-Wl,-rpath,/opt/homebrew/lib"]
    print("[setup.py] Target: macOS")

else:
    # Generic Linux (x86_64 dev machine)
    extra_compile_args += ["-march=native"]
    print("[setup.py] Target: Generic Linux (x86_64)")

# ── Extension ────────────────────────────────────────────────────────────────

ext_module = Pybind11Extension(
    "lidar_invision",
    sources=["src/binding.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    cxx_std=17,
)

# ── Package ──────────────────────────────────────────────────────────────────

setup(
    name="lidar_invision",
    version="1.0.0",
    description="YDLIDAR rolling-buffer Python bindings (pybind11)",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pybind11>=2.11.0",
    ],
    ext_modules=[ext_module],
    cmdclass={
        "build_ext": build_ext,
        "install":   PostInstallCommand,
        "develop":   PostDevelopCommand,
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
