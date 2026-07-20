#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ruff: noqa: F821
# (ruff don't see read variables from release.py)

from setuptools import find_namespace_packages, setup
from os.path import join, dirname


exec(open(join(dirname(__file__), 'kplus', 'release.py'), 'rb').read())  # Load release variables
lib_name = 'kplus'

setup(
    name='kplus',
    version=Release.version,
    description=Release.description,
    long_description=Release.long_desc,
    url=Release.url,
    author=Release.author,
    author_email=Release.author_email,
    classifiers=[c for c in Release.classifiers.split('\n') if c],
    license=Release.license,
    scripts=['kplus-bin'],
    packages=find_namespace_packages(),
    package_dir={'%s' % lib_name: 'kplus'},
    include_package_data=True,
    python_requires='>=' + ".".join(map(str, MIN_PY_VERSION)),
)