#!/usr/bin/env python3
"""
Setup script for Nameplate Detector package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = [
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "numpy>=1.20.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
        "requests>=2.25.0",
        "python-socketio>=5.0.0",
        "transformers>=4.20.0",
        "accelerate>=0.20.0",
        "pydantic>=1.8.0",
    ]

# Development dependencies
dev_requirements = [
    "pytest>=6.0.0",
    "pytest-cov>=2.12.0",
    "pytest-asyncio>=0.18.0",
    "black>=21.0.0",
    "flake8>=3.9.0",
    "isort>=5.9.0",
    "mypy>=0.910",
    "pre-commit>=2.15.0",
    "mkdocs>=1.2.0",
    "mkdocs-material>=7.2.0",
    "mkdocstrings>=0.16.0",
]

setup(
    name="nameplate-detector",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A real-time video streaming application for industrial nameplate detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/nameplate-detector",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/nameplate-detector/issues",
        "Source": "https://github.com/yourusername/nameplate-detector",
        "Documentation": "https://nameplate-detector.readthedocs.io/",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Video",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "pytest-asyncio>=0.18.0",
        ],
        "docs": [
            "mkdocs>=1.2.0",
            "mkdocs-material>=7.2.0",
            "mkdocstrings>=0.16.0",
        ],
        "gpu": [
            "torch>=1.9.0+cu111",
            "torchvision>=0.10.0+cu111",
        ],
    },
    entry_points={
        "console_scripts": [
            "nameplate-detector=nameplate_detector.cli:main",
            "nameplate-api=nameplate_detector.api.server:main",
            "nameplate-predict=scripts.predict_nameplate:main",
            "nameplate-quickstart=scripts.quickstart_camera:main",
        ],
    },
    include_package_data=True,
    package_data={
        "nameplate_detector": [
            "config/*.yaml",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "computer vision",
        "object detection",
        "industrial automation",
        "nameplate detection",
        "real-time processing",
        "video streaming",
        "machine learning",
        "deep learning",
        "PyTorch",
        "OpenCV",
    ],
) 