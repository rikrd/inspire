# inspire's setup.py
from setuptools import setup, find_packages

setup(
    name="inspire",
    packages=find_packages(),
    version="1.0.8",
    description="Helper library to participate in the INSPIRE challenge",
    author="Ricard Marxer",
    author_email="r.marxer@sheffield.ac.uk",
    url="https://github.com/rikrd/inspire",
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
""",
    requires=['numpy', 'docopt', 'grako', 'progressbar_ipython', 'requests'],
    install_requires=['numpy', 'docopt', 'grako', 'progressbar_ipython', 'requests']
)
