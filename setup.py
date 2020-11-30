from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

with open("requirements.txt") as fp:
    install_requires = fp.read().split("\n")

setup(
    name="ecn",
    version="0.3.0",
    description="Event Convolutional Networks for tensorflow",
    url="https://github.com/jackd/ecn.git",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=find_packages(),
    requirements=install_requires,
    zip_safe=True,
)
