from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).absolute().parent

long_description = (here / Path('README.md')).read_text()

_version = {}
exec((here / Path('amarl/_version.py')).read_text(), _version)

setup(
    name='amarl',
    version=_version['__version__'],
    description='Implementation of Argumentative MARL',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/SwamyDev/amarl',
    author='Bernhard Raml',
    packages=find_packages(include=['amarl', 'amarl.*']),
    install_requires=["ray[rllib]", "torch"],
    extras_require={"test": ['pytest', 'pytest-cov', 'gym-quickcheck']},
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.6'
)
