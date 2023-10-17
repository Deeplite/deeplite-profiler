# -*- coding: utf-8 -*-
from setuptools import setup, find_namespace_packages
from setuptools.command.build_py import build_py as _build_py
import os
import sysconfig

try:
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

EXCLUDE_FILES = []

class build_py(_build_py):

    def find_package_modules(self, package, package_dir):
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
        modules = super().find_package_modules(package, package_dir)
        filtered_modules = []
        for (pkg, mod, filepath) in modules:
            if os.path.exists(filepath.replace('.py', ext_suffix)):
                continue
            filtered_modules.append((pkg, mod, filepath, ))
        return filtered_modules


def get_ext_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            if os.path.splitext(filename)[1] != '.py':
                continue

            file_path = os.path.join(root, filename)
            if file_path in exclude_files:
                continue

            paths.append(file_path)
    return paths

def get_c_ext_modules():
    if use_cython:
        return cythonize(
                    get_ext_paths('deeplite', EXCLUDE_FILES),
                    compiler_directives={'language_level': 3}
                )
    else:
        return []

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open('LICENSE') as f:
    license = f.read()

TORCH_RANGE = "torch>=1.4, <=1.13"
TFLITE = "tflite==1.14; python_version <= '3.7.10'"
DATA_PKGS = ["pandas<=1.4.4", "matplotlib"]
EXTRAS_REQUIRE = {
    'torch': [TORCH_RANGE, "ptflops==0.6.2", *DATA_PKGS],
    'tf-gpu': ["tensorflow-gpu==1.14; python_version <= '3.7.10'", "protobuf==3.19.*", *DATA_PKGS],
    'tf': [TFLITE, "tensorflow==1.14; python_version <= '3.7.10'", "protobuf==3.19.*", *DATA_PKGS],
    'all': [TFLITE, TORCH_RANGE, "ptflops==0.6.2", "tensorflow==1.14; python_version <= '3.7.10'", "protobuf==3.19.*", *DATA_PKGS],
    'all-gpu': [TORCH_RANGE, "ptflops==0.6.2", "tensorflow-gpu==1.14; python_version <= '3.7.10'", "protobuf==3.19.*", *DATA_PKGS],
}

setup(
    name='deeplite-profiler',
    version='1.2.5',
    description='Profiler for deep learning models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Deeplite',
    author_email="support@deeplite.ai",
    url="https://github.com/Deeplite/deeplite-profiler",
    license='Apache 2.0',
    packages=find_namespace_packages(exclude=('tests*', 'docs',)),
    package_data={
    'deeplite': ['deeplite-profiler.sig'],
    },
    install_requires=["setuptools<65.6.0", "numpy==1.20.0"],
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
    #include_package_data=True,
    ext_modules=get_c_ext_modules(),
    cmdclass={
        'build_py': build_py
    }
)
