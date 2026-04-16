from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext # pip install pybind11

ext_module = Pybind11Extension(
    "inference_engine_cpp",
    ["src/model/cpp_inference/inference_engine.cpp"],
    include_dirs=[],  
    library_dirs=[],  
    libraries=["tensorflow-lite"],
    extra_compile_args=["-std=c++17", "-O3", "-march=native"], # -O3 and march=native are key for Pi 5 speed
)

setup(
    name="cv_invision",
    version="1.0.0",
    # This says: "The package 'cv_invision' is actually found in the 'src' directory"
    package_dir={"": "src"},
    # This finds all sub-packages inside the 'src' directory
    packages=find_packages(where="src"),
    install_requires=[
        # Core Math & Computer Vision
        "numpy>=1.23.0",
        "opencv-python>=4.6.0",
        "pillow>=9.0.0",
        "tensorflow>=2.15.0,<2.18",
        "protobuf>=3.20.3",

        
        # AI & Tracking
        "ultralytics>=8.0.0",
        
        # Dataset & Analysis (Found in src/dataset/utils)
        "pandas>=1.5.0",      # Used in analyze_dataset.py
        "matplotlib>=3.5.0",  # Used in preview_image.py and analyze_dataset.py
        "pyyaml>=6.0",        # Used in remap_classes.py and list_classes.py
        
        # Utilities (Found in src/utils)
        "tqdm>=4.64.0",       # Used for progress bars in image_compress.py and build.py
        "gdown>=4.6.0",       # Used in download_samples.py to get data from Google Drive
    ],
    cmdclass={"build_ext": build_ext},
)


