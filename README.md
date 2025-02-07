# SN-Graph: a graph skeletonisation algorithm.

A Python implementation of an SN-Graph skeletonisation algorithm. Based on the article *SN-Graph: a Minimalist 3D Object Representation for Classification* [arXiv:2105.14784](https://arxiv.org/abs/2105.14784). Currently works only for 2D images.


![Example of a binary image and the skeletal graph](/assets/horse_graphs.png "SN-graph generated out of an scikit-image's horse image.")

## Description

SN-Graph works by:

1. Creating vertices as centres of spheres inscribed in the image, where one balances the size of the spheres with their coverage of the shape, and pariwise distances from one another.
3. Adding edges between the neighbouring spheres, subject to a few common-sense criteria.

The resulting graph serves as a lightweight 1-dimensional representation of the original image, potentially useful for further analysis.

## Basic Usage

```python
import numpy as np
import sn_graph as sn

# Create a simple square image
img = np.zeros((100, 100))
img[40:60, 40:60] = 1  # Create a square region

# Generate the SN graph
centers, edges = sn.create_SN_graph(
    img,
    max_num_vertices=10,
    edge_threshold=1.0
)

```

## Key Parameters

- `max_num_vertices`: Maximum number of vertices in the graph
- `max_edge_length`: Maximum allowed edge length
- `edge_threshold`: Threshold for determining what portion of an edge must be contained within the shape
- `minimal_sphere_radius`: Minimum radius allowed for spheres

## Authors
- Tomasz Prytuła (<tomasz.prytula@alexandra.dk>)
