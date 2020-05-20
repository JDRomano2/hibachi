#!/usr/bin/env python

from distutils.core import setup

setup(
    name='Hibachi',
    version='1.0-alpha',
    description='Data simulation software that creates data sets with particular characteristics',
    author='Pete Schmitt and Joseph D. Romano',
    author_email='joseph.romano@pennmedicine.upenn.edu',
    url='https://github.com/EpistasisLab/hibachi',
    python_requires='>=3.7',
    install_requires=[
        'deap',
        'scikit-mdr',
        'pygraphviz'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Environment :: Console"
    ]
)
