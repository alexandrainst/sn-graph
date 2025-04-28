# SN-Graph: a graph skeletonisation algorithm.



<div align="center">

[![build](https://github.com/alexandrainst/sn-graph/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/alexandrainst/sn-graph/actions/workflows/build-and-test.yml)
[![Coverage Report](assets/coverage.svg)](https://github.com/alexandrainst/sn-graph/tree/main/tests)
[![PyPI - Version](https://img.shields.io/pypi/v/sn-graph)](https://pypi.org/project/sn-graph/)
[![License](https://img.shields.io/github/license/alexandrainst/sn-graph)](https://github.com/alexandrainst/sn-graph/blob/main/LICENSE)

</div>




A Python implementation of an SN-Graph skeletonisation algorithm. Based on the article *SN-Graph: a Minimalist 3D Object Representation for Classification* [arXiv:2105.14784](https://arxiv.org/abs/2105.14784).


![Example of a binary image and the skeletal graph](https://raw.githubusercontent.com/alexandrainst/sn-graph/main/assets/horse_graph.png "SN-graph generated out of an scikit-image's horse image.")

## Description

Given a binary image/volume representing a shape, SN-Graph works by:

1. Creating vertices as centres of spheres inscribed in the shape, where one balances the size of the spheres with their coverage of the shape, and pariwise distances from one another.
3. Adding edges between the neighbouring spheres, subject to a few common-sense criteria.

The resulting graph serves as a lightweight 1-dimensional representation of the original image, potentially useful for further analysis.

## Installation

```bash
pip install sn-graph
```
or

```bash
poetry add sn-graph
```

## Basic Usage

See notebooks [demo_sn-graph](https://github.com/alexandrainst/sn-graph/blob/main/notebooks/demo_sn-graph.ipynb) and [3d_demo](https://github.com/alexandrainst/sn-graph/blob/main/notebooks/3D_demo.ipynb) for 2D and 3D demo, respectively. Notebook [mnist_classification](https://github.com/alexandrainst/sn-graph/blob/main/notebooks/mnist_classification.ipynb) has some good stuff too!

```python
import numpy as np
import sn_graph as sn

# Create a simple square image
img = np.zeros((256, 256))
img[20:236, 20:236] = 1  # Create a square region

# Generate the SN graph
centers, edges, sdf_array = sn.create_sn_graph(
    img,
    max_num_vertices=15,
    minimal_sphere_radius=1.0,
    return_sdf=True
)

import matplotlib.pyplot as plt

#Draw graph on top of the image and plot it
graph_image=sn.draw_sn_graph(centers, edges, sdf_array, background_image=img)

plt.imshow(graph_image)
plt.show()
```
<img src="https://raw.githubusercontent.com/alexandrainst/sn-graph/main/assets/square_readme.png" alt="SN-Graph drawn on top of the square" width="500">

## Key Parameters

- `max_num_vertices`: Maximum number of vertices in the graph
- `max_edge_length`: Maximum allowed edge length
- `edge_threshold`: Threshold for determining what portion of an edge must be contained within the shape
- `minimal_sphere_radius`: Minimum radius allowed for spheres
- `edge_sphere_threshold`: Threshold value for deciding how close can an edge be to a non-enpdpoint spheres
- `return_sdf`: Whether to return signed distance field array computed by the algorithm (neccessary to extract radii of spheres)

## Authors
- Tomasz Prytuła (<tomasz.prytula@alexandra.dk>)
