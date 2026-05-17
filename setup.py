from setuptools import setup, find_packages

setup(
    name="invision_central_control",
    version="1.0.0",
    description="Master Orchestrator package for InVision Autonomous Tracking System",
    author="alejerw",
    python_requires=">=3.11",
    # Installs central_control alongside its core sibling requirements
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Refers to system installation requirements
        "numpy>=2.0.0",
        "opencv-python-headless>=4.9.0",
    ],
    entry_points={
        "console_scripts": [
            "invision-run=central_control.app:main",
        ],
    },
)
