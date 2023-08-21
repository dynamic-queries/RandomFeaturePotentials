### RandomFeaturePotentials
Fits potential energy surfaces of different molecules, materials using [Random feature models](https://arxiv.org/abs/2005.10224).

The support for RFMs are randomly sampled along the input domain. Here's RFMs in action for a two-output sinusoidal function, initially sampled uniformly and in the second case with weighted sampling.

![](https://github.com/dynamic-queries/RandomFeaturePotentials/blob/main/test/approx_test/Uniform_multiple_sampling.svg)

![](https://github.com/dynamic-queries/RandomFeaturePotentials/blob/main/test/approx_test/FiniteDifference_multiple_sampling.svg)
#### Literature

##### Covariance Kernels
1. Chapter 4 of [GP Book](https://gaussianprocess.org/gpml/chapters/RW.pdf)
2. [Hypergraph GP](https://arxiv.org/abs/2106.01982)
3. [GP on Riemannian manifolds](https://arxiv.org/abs/2006.10160)
4. A nice thesis on [similarity measures on hypergraphs](https://we.vub.ac.be/sites/default/files/Thesis_Filip_Moons.pdf)

##### Invariants
5. [Deep Sets](https://arxiv.org/abs/1703.06114)
6. [Augmented Similarity Descriptors - Weinan's paper](https://arxiv.org/abs/1805.09003)
  
##### Ecosystems
7. [SGDML](www.sgdml.org) 
