import pathlib
from setuptools import setup,  find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="openwind",
    version="0.11.1",
    description="Open source library to design wind instruments",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://files.inria.fr/openwind/docs",
    author="Makutu",
    author_email="openwind-contact@inria.fr",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
            'numpy',
            'scipy',
            'matplotlib',
            'h5py',
    ],
    entry_points={
        "console_scripts": [
            "",
        ]
    },
)
