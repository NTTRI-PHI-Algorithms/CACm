from setuptools import setup, find_packages

setup(
    name="CACm",
    version="1.22",
    description="Chaotic Amplitude Control with momentum",
    author="Timothee Leleu",
    author_email="timothee.leleu@ntt-research.com",
    packages=find_packages(),
    install_requires=[
        "numpy==1.26.0",
        "pandas==2.0.3",
        "matplotlib==3.7.2",
        "torch==2.0.1",
        ],
)