from setuptools import setup, find_packages

setup(
    name="cv_invision",
    version="1.0.0",
    # This says: "The package 'cv_invision' is actually found in the 'src' directory"
    package_dir={"": "src"},
    # This finds all sub-packages inside the 'src' directory
    packages=find_packages(where="src"),
    install_requires=[
        "ultralytics",
        "opencv-python",
        "pillow",
    ],
)


