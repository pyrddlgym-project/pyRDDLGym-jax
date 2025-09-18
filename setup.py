# This file is part of pyRDDLGym.

# pyRDDLGym is free software: you can redistribute it and/or modify
# it under the terms of the MIT License as published by
# the Free Software Foundation.

# pyRDDLGym is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# MIT License for more details.

# You should have received a copy of the MIT License
# along with pyRDDLGym. If not, see <https://opensource.org/licenses/MIT>.

from setuptools import setup, find_packages

from pathlib import Path
long_description = (Path(__file__).parent / "README.md").read_text()

setup(
      name='pyRDDLGym-jax',
      version='2.7',
      author="Michael Gimelfarb, Ayal Taitler, Scott Sanner",
      author_email="mike.gimelfarb@mail.utoronto.ca, ataitler@gmail.com, ssanner@mie.utoronto.ca",
      description="pyRDDLGym-jax: automatic differentiation for solving sequential planning problems in JAX.",
      long_description=long_description,
      long_description_content_type='text/markdown',
      license="MIT License",
      url="https://github.com/pyrddlgym-project/pyRDDLGym-jax",
      packages=find_packages(),
      install_requires=[
          'pyRDDLGym>=2.3',
          'tqdm>=4.66',
          'jax>=0.4.12',
          'optax>=0.1.9',
          'dm-haiku>=0.0.10',
          'tensorflow-probability>=0.21.0'
      ],
      extras_require={
          'extra': ['bayesian-optimization>=2.0.0', 'rddlrepository>=2.0'],
          'dashboard': ['dash>=2.18.0', 'dash-bootstrap-components>=1.6.0']
      },
      python_requires=">=3.9",
      package_data={'': ['*.cfg', '*.ico']},
      include_package_data=True,
      entry_points={ 
          'console_scripts': [ 'jaxplan=pyRDDLGym_jax.entry_point:main'],
      },
      classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
