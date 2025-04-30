# Installation and basic usage


## Installation

```bash
pip install sn-graph
```
or

```bash
poetry add sn-graph
```
## Basic usage

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


## Tutorials
* [demo_sn-graph](https://github.com/alexandrainst/sn-graph/blob/main/notebooks/demo_sn-graph.ipynb) A notebook with a 2D demo
* [3d_demo](https://github.com/alexandrainst/sn-graph/blob/main/notebooks/3D_demo.ipynb) A ntoebook with 3d demo
* [mnist_classification](https://github.com/alexandrainst/sn-graph/blob/main/notebooks/mnist_classification.ipynb) Mnist classification with SN-gaph nad Graph Neural Networks
