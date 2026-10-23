from pathlib import Path

from setuptools import find_packages, setup

version = "0.4.0"
root = Path(__file__).parent

setup(
    name="neura_command",
    version=version,
    packages=find_packages(
        where="src",
        include=["app", "app.*", "neura_command", "neura_command.*"],
    ),
    package_dir={"": "src"},
    install_requires=["numpy>=1.20.0"],
    extras_require={
        "cuda12": ["cupy-cuda12x>=12"],
        "cuda13": ["cupy-cuda13x>=14"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "neura_command=neura_command.cli:main",
        ]
    },

    # Metadata about the package
    author="Adam Djellouli",
    author_email="addjellouli1@gmail.com",
    description="NeuraCommand is a simplistic tool for creating, training, and deploying neural networks. It provides a seamless interface for handling complex neural network architectures, including perceptrons with multiple layers. With NeuraCommand, users can efficiently load data, train models, and use them for predictions.",

    # Use a stable path so packaging works outside the repo root.
    long_description=(root / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/djeada/NeuraCommand",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Add additional metadata as needed
)
