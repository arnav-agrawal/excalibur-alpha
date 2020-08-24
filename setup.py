#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:22:53 2020

@author: arnav
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="excalibur.eng", # Replace with your own username
    version="0.0.1",
    author="Arnav Agrawal",
    author_email="aa687@cornell.edu",
    description="A package to calculate atomic and molecular cross sections.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnav-agrawal/Cthulhu",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.6',
    install_requires = [
        "numpy", "scipy", "matplotlib", "numba", "requests", "bs4", "tqdm"
    ]
)