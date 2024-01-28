#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="weightbridge",
    version="0.0.3",
    python_requires=">=3",
    description="Library to map (deep learning) model weights between different model implementations.",
    author="Florian Fervers",
    author_email="florian.fervers@gmail.com",
    url="https://github.com/fferflo/weightbridge",
    packages=find_packages(),
    license="MIT",
    include_package_data=True,
    install_requires=[
        "numpy",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
