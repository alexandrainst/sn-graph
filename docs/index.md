# Introduction




## What is SN-Graph?

`sn-graph` is a Python library containing an implementation of the SN-Graph algorithm.

The SN-Graph is a `graph skeletonisation algorithm`, that is, an algorithm that takes a 2D/3D shape (e.g. a volume) and computes a graph skeleton of that shape. The skeleton is a lightweight, 1-dimensional representation that preserves the rough geometric structure of the shape. As such, it can be used for further processing, and various ML tasks, such as classification, regression, etc.

The goal of `sn-graph` is to provide the first Python implementation of the algorithm, together with functions to easily visualise the effect either as a volume (numpy array) or as an interactive 3d scene via trimesh.


## About the algorithm

The algorithm was introduced in the article *SN-Graph: a Minimalist 3D Object Representation for Classification* [arXiv:2105.14784](https://arxiv.org/abs/2105.14784).
The algorithm takes as an input a binary (or a monochromatic) image/volume, where 0 values represent boudnary contour of the object. It consists of the following three steps:

  - compute the Signed Distance Field
  - add vertices, seen as centres of spheres inscribed in the image (one uses SDF to query cooridnates for the spheres radii). The order of adding spheres is a minimax-style queue, where one balances maximising the distances between the points (like in Furthest Point Sampling algorithm), and the size of the introcuded spheres. **The strength of this approach lies in the fact that even with few vertices, it describes the shape well.**
  - add edges between the spheres. Edges are added bewteen the chosen vertices based ina caouple criteria, like e.g. an edge has to be mostly contained within the shape, or an edge should not cross any spheres other than its endpoints.
