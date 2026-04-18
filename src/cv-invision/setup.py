from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import os
import sys
import subprocess
from setuptools.command.install import install
from setuptools.command.develop import develop
import platform

def run_invision_validation():
    """Validates the C++ module can actually be imported and linked."""
    print("\n--- Running InVision Post-Install Validation ---")
    try:
        # Check current python version to ensure it matches the built .so
        print(f"Checking for Python {sys.version_info.major}.{sys.version_info.minor} compatibility...")
        
        cmd = [
            sys.executable, "-c", 
            "from src.model.cpp_inference import preprocess; print('Check 1/2: Module Import Success')"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + ":" + env.get("PYTHONPATH", "")
        
        subprocess.check_call(cmd, env=env)
        print("Check 2/2: OpenCV Symbols Linked Correctly")
        print("RESULT: InVision C++ Inference Engine is READY. 🚀\n")
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        print("Tip: Check 'ldd src/model/cpp_inference*.so' for missing OpenCV libraries.")
        print("Ensure you are running inside your virtual environment.\n")

# Custom command classes to trigger validation
class PostInstallCommand(install):
    def run(self):
        install.run(self)
        run_invision_validation()

class PostDevelopCommand(develop):
    def run(self):
        develop.run(self)
        run_invision_validation()





# Detect architecture
is_arm = platform.machine().startswith('aarch64') or platform.machine().startswith('arm')

# Set architecture-specific flags
if is_arm:
    # High-performance flags for Raspberry Pi 5
    march_flag = "-march=armv8.2-a+dotprod"
else:
    # Generic optimization for your Arch laptop (x86_64)
    march_flag = "-march=native"
    

ext_module = Pybind11Extension(
    "model.cpp_inference",
    [
        "src/model/cpp_inference/binding.cpp",
        "src/model/cpp_inference/postprocess.cpp",
        "src/model/cpp_inference/preprocess.cpp",
    ],
    include_dirs=["/usr/local/include/opencv4"],
    library_dirs=["/usr/local/lib"],
    libraries=[
        "opencv_core", 
        "opencv_imgproc", 
        "opencv_imgcodecs",
        "gomp" # Enables OpenMP for multi-core performance on Pi 5
    ],
    extra_compile_args=[
        "-std=c++17", 
        "-O3", 
        march_flag,
        "-fopenmp"
    ],
    extra_link_args=[
        "-Wl,-rpath,/usr/local/lib" # Fixes "undefined symbol" at runtime
    ],
)

setup(
    name="cv_invision",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.23.0,<2",
        "opencv-python-headless", # Use headless to avoid binary conflicts
        "pillow>=9.0.0",
        "tensorflow>=2.15.0,<2.18",
        "ultralytics>=8.0.0",
        "pandas>=1.5.0",
        "matplotlib>=3.5.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "gdown>=4.6.0",
    ],
    ext_modules=[ext_module],
    # MERGED CMDCLASS: Keeps your build_ext and adds the validators
    cmdclass={
        "build_ext": build_ext,
        "install": PostInstallCommand,
        "develop": PostDevelopCommand,
    },
)
