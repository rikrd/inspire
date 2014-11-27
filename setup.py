# inspire's setup.py
from distutils.core import setup

setup(
    name="inspire",
    packages=["inspire"],
    version="1.0.0",
    description="Helper library to participate in the INSPIRE challenge",
    author="Ricard Marxer",
    author_email="r.marxer@sheffield.ac.uk",
    url="http://www.ricardmarxer.com/research/inspire_challenge",
    download_url="http://chardet.feedparser.org/download/python3-chardet-1.0.1.tgz",
    keywords=["intelligibility", "speech", "evaluation"],
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    long_description="""\
INSPIRE Challenge library
-------------------------------------

This library includes a set of functions to help in the participation of the INSPIRE challenge.

The functions included in this module include:
 - loading the dataset
 - loading the lexicon
 - creating a participation
 - submitting a participation

Additionally it contains functions to create, train and use speech recognition models using HTK.
"""
)
