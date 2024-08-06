from setuptools import setup

setup(
    name="opt_dynamiqs",
    version="0.1",
    packages=[
        "opt_dynamiqs",
    ],
    description="optimal control wrapper of dynamiqs",
    long_description=open("README.md").read(),
    author="Daniel Weiss",
    author_email="daniel.kamrath.weiss@gmail.com",
    url="https://github.com/dkweiss31/opt-dynamiqs",
    install_requires=[
        "numpy",
        "scipy",
        "h5py",
        "dynamiqs",
        "jaxtyping",
        "jax",
        "optax",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
