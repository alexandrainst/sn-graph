# Introduction

![Animation of adding spheres to the graph](https://raw.githubusercontent.com/alexandrainst/sn-graph/main/assets/horse_animation.gif)

## What is SN-Graph?

`sn-graph` is a Python library implementing the SN-Graph (Sphere Node Graph) algorithm for graph skeletonization. This algorithm converts 2D/3D shapes into lightweight, 1-dimensional graph skeletons while preserving their rough geometric structure.

The library provides:

  - The Python implementation of the SN-Graph algorithm.
  - Visualization functions for both volumetric (numpy array) representations and interactive 3D scenes via trimesh.


## About the algorithm

Originally introduced in [SN-Graph: a Minimalist 3D Object Representation for Classification](https://arxiv.org/abs/2105.14784), the algorithm processes binary/monochromatic images or volumes through three steps:

1. Compute the Signed Distance Field (SDF) of the object contour.
2. Add vertices, seen as centers of spheres inscribed in the shape, where SDF is used to get the radii of the spheres. The order of adding spheres is a minimax-style queue that balances pairwise distances between the spheres and their radii.
3. Place edges between vertices based on criteria such as 'an edge has to be mostly contained within the shape', or 'an edge should not cross any spheres other than its endpoints'.

## Why SN-Graph?

SN-Graph offers several advantages over traditional skeletonization methods:

- **Efficient representation:** With few vertices, SN-Graph captures the essential structure of complex shapes.
- **Highly customizable:** The algorithm provides multiple parameters to fine-tune the skeleton generation process according to specific requirements.
- **Abstract graph output:** Unlike classical skeletonization approaches that produce pixel-based segmentations, SN-Graph generates an abstract graph representation, making it directly amenable to e.g. Graph Machine Learning algorithms.

See the [Installation and basic usage](installation.md) section to begin using SN-Graph.
