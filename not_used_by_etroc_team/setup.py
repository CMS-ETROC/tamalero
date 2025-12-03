#!/usr/bin/env python3
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# get the dependencies and installs
with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    all_reqs = f.read().split("\n")

install_requires = [x.strip() for x in all_reqs if "git+" not in x]
dependency_links = [
    x.strip().replace("git+", "") for x in all_reqs if x.startswith("git+")
]

setup(
    name='tamalero',
    version='0.1.0',
    description="CMS MTD ETL test stand software",
    url="https://gitlab.cern.ch/cms-etl-electronics/module_test_sw/",
    packages=find_packages(exclude=["docs", "tests*", "examples"]),
    install_requires=install_requires,
    dependency_links=dependency_links,
)
