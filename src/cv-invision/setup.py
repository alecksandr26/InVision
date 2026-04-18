from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_module = Pybind11Extension(
    "model.postprocess_cpp",
    ["src/model/cpp_inference/postprocess.cpp"],
    include_dirs=[],
    library_dirs=[],
    libraries=[],                     # no tensorflow-lite
    extra_compile_args=["-std=c++17", "-O3", "-march=native"],
)

setup(
    name="cv_invision",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.23.0,<2",           # pinned to <2 for tflite-runtime compatibility
        "opencv-python>=4.6.0",
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
    cmdclass={"build_ext": build_ext},
)
