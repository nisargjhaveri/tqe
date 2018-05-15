#!/usr/bin/env python

from setuptools import setup

setup(
    name='tqe',
    version='0.0',
    description='Translation Quality Estimation',
    author='Nisarg Jhaveri',
    author_email='nisargjhaveri@gmail.com',
    url='https://github.com/nisargjhaveri/tqe',
    packages=['tqe', 'tqe.utils'],
    scripts=['tqecli.py'],
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "polyglot",
        "regex",
        "joblib",
        "kenlm==git.master",
        "fastText==git.master",
        # keras
        "keras>=2.1.6",
        "h5py",
    ],
    dependency_links=[
        "https://github.com/kpu/kenlm/archive/master.zip#egg=kenlm-git.master",
        "https://github.com/facebookresearch/fastText/archive/master.zip#egg=fastText-git.master",
    ]
)
