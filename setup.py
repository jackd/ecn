import os
from pathlib import Path

from setuptools import find_packages, setup

with open("requirements.txt") as fp:
    install_requires = fp.read().split("\n")


def glob_fix(package_name, glob):
    package_path = Path(os.path.join(os.path.dirname(__file__), package_name)).resolve()
    return [str(path.relative_to(package_path)) for path in package_path.glob(glob)]


setup(
    name="ecn",
    version="0.3.0",
    description="Event Convolutional Networks for tensorflow",
    url="https://github.com/jackd/ecn.git",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={"ecn": glob_fix("ecn", "configs/**/*.gin")},
    requirements=install_requires,
    zip_safe=True,
)
