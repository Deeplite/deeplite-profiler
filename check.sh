#!/bin/sh
pip install --upgrade setuptools wheel twine
rm -rf dist
python setup.py sdist
python setup.py bdist_wheel
twine check dist/*