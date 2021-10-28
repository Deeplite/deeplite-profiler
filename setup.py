# -*- coding: utf-8 -*-
from setuptools import setup, find_namespace_packages

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('LICENSE') as f:
    license = f.read()

TORCH_RANGE = "torch>=1.4, <=1.8.1"
EXTRAS_REQUIRE = {
    'torch': [TORCH_RANGE, "ptflops==0.6.2"],
    'tf-gpu': ["tensorflow-gpu==1.14; python_version <= '3.7.10'"],
    'tf': ["tensorflow==1.14; python_version <= '3.7.10'"],
    'all': [TORCH_RANGE, "ptflops==0.6.2", "tensorflow==1.14; python_version <= '3.7.10'"],
    'all-gpu': [TORCH_RANGE, "ptflops==0.6.2", "tensorflow-gpu==1.14; python_version <= '3.7.10'"],
}

setup(
    name='deeplite-profiler',
    version='1.1.12',
    description='Profiler for deep learning models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Deeplite',
    author_email="support@deeplite.ai",
    url="https://github.com/Deeplite/deeplite-profiler",
    license='Apache 2.0',
    packages=find_namespace_packages(exclude=('tests*', 'docs',)) + ['.'],
    install_requires=["numpy>=1.17"],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',        
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'License :: OSI Approved :: Apache Software License',
        'Environment :: Console'
    ],
    keywords='profiler deep_neural_network deep_learning torch deeplite metrics',
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
)
