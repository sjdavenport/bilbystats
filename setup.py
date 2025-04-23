from distutils import sysconfig
from setuptools import setup, Extension, find_packages
import os
import sys
import setuptools
from copy import deepcopy

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='bilbystats',
    install_requires=[
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'transformers'
    ],
    version='0.0.3',
    license='MIT',
    author='Samuel DAVENPORT',
    download_url='https://github.com/sjdavenport/bilbystats/',
    author_email='samuel.davenport@math.univ-toulouse.fr',
    url='https://github.com/sjdavenport/bilbystats/',
    long_description=long_description,
    description='Python Packages of functions for performing stats for bilby',
    keywords='LLMs',
    packages=find_packages(),
    python_requires='>=3',
)
