""" Necessary lib installer """
from setuptools import find_packages, setup


with open('requirements.txt', 'r', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='HW1_ready_project',
    packages=find_packages(),
    version="0.1.0",
    description='''
    'Production ready' project for MLOps course by Trenev Ivan, ML-21.
    ''',
    author='Trenev Ivan',
    license='MIT',
    install_requires=required,
)
