#!/usr/bin/env python3

# #############################################################################
# setup.py
# ========
# Author : Sepand KASHANI [kashani.sepand@gmail.com]
# #############################################################################

"""
Setup script.
"""

import sys

from setuptools import setup


if sys.version_info < (3, 7):
    # pbr not supported: audio_tools will be installed without any version info.
    kwargs = dict()
else:
    kwargs = dict(setup_requires=["pbr"], pbr=True)

setup(**kwargs)
