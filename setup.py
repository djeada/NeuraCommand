from setuptools import find_packages, setup

version = "0.4.0"

setup(
    name="neura_command",
    version=version,
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy>=1.20.0"],

    # Metadata about the package
    author="Adam Djellouli",
    author_email="addjellouli1@gmail.com",
    description="NeuraCommand is a simplistic tool for creating, training, and deploying neural networks. It provides a seamless interface for handling complex neural network architectures, including perceptrons with multiple layers. With NeuraCommand, users can efficiently load data, train models, and use them for predictions.",

    # Use context manager for reading the long description
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/djeada/NeuraCommand",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # Add additional metadata as needed
)
