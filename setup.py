import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="excalibur.exo",
    version="0.1.0",
    author="Ryan MacDonald, Arnav Agrawal",
    author_email="rmacdonald@astro.cornell.edu, aa687@cornell.edu",
    description="A python package to calculate atomic and molecular cross sections for exoplanet atmospheres.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnav-agrawal/excalibur-alpha",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.6',
    install_requires = [
        "numpy", "scipy", "matplotlib", "numba", "requests", "bs4", "tqdm", "pandas", "h5py"
    ],
    zip_safe=False
)
