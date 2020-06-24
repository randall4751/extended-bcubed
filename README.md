# Extended B-Cubed Clustering Metric

The BCubed metric has been shown to be a useful tool for comparing clustering results. See the excellent paper ["A comparison of extrinsic clustering evaluation metrics bsed on formal constraints", *Amigo E, Gonzalo J, Artiles J et al.*](https://www.researchgate.net/publication/225548032)

This code contains a vectorized (numpy) version of the extended version of the B-Cubed metric which handles the multiplicity case where items may be associated with more than one category.

The unit tests include examples of the simple case (items belong to one and only one category) from the reference paper's Figure 11.

    Note - It appears that 3 of the 4 example pairs in the Figure have incorrect results.

The unit tests also include the multiplicity examples from Figures 16 through 21.

Running the code without unit tests executes an extended example using two adjacency matrices representing two 23-node graphs. Maximal cliques are extracted from the graphs. The cliques are used as categories and inferred clusters and compared using the extended BCubed metric.

Although the extended BCubed algorithm also works for the simple case, a separate BCubed algorithm for the simple case is included in the code.

The repository also contains a Jupyter Lab notebook that steps through the multiplicity BCubed algorithm.
