from setuptools import setup, find_packages

setup(name='gym_CTgraph',
      version='1.0',
      install_requires=['gymnasium', 'torch', 'torchvision', 'scikit-image'],  # And any other dependencies foo needs
      packages=find_packages(),
      author="Andrea Soltoggio, Eseoghene Iwhiwhu, Christos Peridis, Pawel Ladosz, Jeff Dick",
      author_email="a.soltoggio@lboro.ac.uk",
      description="This is the implementation for the configurabe tree graph (CT-graph)",
      license="Copyright (C) 2019-2021 Andrea Soltoggio, Pawel Ladosz, Eseoghene Iwhiwhu, Jeff Dick. This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.",
      keywords="Deep reinforcement learning, dynamic rewards, continual learning, adaptation, partially observable Markov decision problems, POMDP",
      url="https://github.com/soltoggio/CT-graph",
      project_urls={
        "Bug Tracker": "",
        "Documentation": "",
        "Source Code": "https://github.com/soltoggio/CT-graph",
    }
)
