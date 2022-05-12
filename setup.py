import os
import setuptools

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(name='mac',
version='0.1',
description='Fast algebraic connectivity maximization',
url='#',
author='keevindoherty',
install_requires=requirements,
author_email='kdoherty@mit.edu',
packages=setuptools.find_packages(),
zip_safe=False)
