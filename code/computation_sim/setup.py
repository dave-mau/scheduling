from setuptools import find_packages, setup

setup(
    name="computation_sim",
    version="0.0",
    packages=find_packages(include=["computation_sim", "computation_sim.*"]),
    author="David Mauderli",
    author_email="davidmauderli@gmail.com",
    description="A package for simulating a distributed system with sensors and computation nodes.",
    url="https://github.com/davidmauderli/computation_sim",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
