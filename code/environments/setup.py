from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="computation_sim_gym",
    version="0.1.0",
    packages=find_packages(include=["computation_sim_gym", "computation_sim_gym.*"]),
    install_requires=[
        "computation_sim @ file://{}#egg=computation_sim".format(
            str(Path(__file__).absolute().parents[1] / "computation_sim")
        )
    ],
    license="MIT",
    description="A package with environments following the Gym API for simulating distributed systems with sensors and computation nodes.",
    author="David Mauderli",
    author_email="davidmauderli@gmail.com",
    url="https://github.com/davidmauderli/scheduling",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
