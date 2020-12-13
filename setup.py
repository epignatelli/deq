#!/usr/bin/env python

from distutils.core import setup

setup(
    name="deq-jax",
    version="0.0.1",
    description="A jax implementation of Deep Equilibrium Models",
    author="Eduardo Pignatelli",
    author_email="edu.pignatelli@gmail.com",
    url="https://github.com/epignateli/deq",
    packages=["deq"],
    install_requires=[
        "pytest",
        "jaxlib",
        "jax",
    ],
)
