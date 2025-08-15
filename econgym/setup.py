from setuptools import setup, find_packages

setup(
    name="econgym",
    version="0.1",
    packages=find_packages(exclude=["envs", "envs.*"]),
    install_requires=[
        "numpy>=1.23",
        "torch>=1.12",
        "matplotlib>=3.5",
        "gymnasium>=0.28",
    ],
) 