# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='DrugMarket',
    version='0.1.0',
    description='Clinical trial based market data for evaluation of pharmaceutical stocks',
    long_description=readme,
    author='Shane Neeley',
    url='https://github.com/Shane-Neeley/DrugMarket',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
