import numpy as np
import networkx as nx   # used only for example graph cliques as categories
import unittest

def f_score(p,r,alpha=0.5):
    '''
        Compute weighted f-score
            alpha = 0.5 is equal weighting between precision and recall
            alpha > 0.5 weights precision over recall
            alpha < 0.5 weights recall over precision
    '''
    return (r*p)/(alpha*r + (1.0 - alpha)*p)

def b_cubed_simple(clusters, categories):
    '''
        Compute the extended b-cubed metric

        clusters  : a list of item arrays representing computed item clusters
        categories: a list of item arrays representing the given, true item categories

        Item labels are assumed to a set of contiguous integers in the range [0..N-1] where N
        is the number of items

        return: precision, recall

        The algorithm is taken from "A comparison of extrinsic clustering evaluation metrics based on
        formal constraints", Amigo E, Gonsalo J, Artiles J, et al. May 11, 2009, https://www.researchgate.net/publication/225548032

    '''
    #
    #   Step 1 - compute inverse maps
    #
    max_item = -1
    for cluster in clusters:
        for item in cluster:
            max_item = max(max_item, item)
    n_items = max_item+1

    cluster_inverse_map = np.empty(n_items, dtype=np.int32)
    for i, cluster in enumerate(clusters):
        for item in cluster:
            cluster_inverse_map[item] = i

    category_inverse_map = np.empty(n_items, dtype=np.int32)
    for i, category in enumerate(categories):
        for item in category:
            category_inverse_map[item] = i

    #
    #   Step 2 - create matrices of (e,e') where inverse_map(e) == inverse_map(e')
    #
    cluster_matrix = np.array([cluster_inverse_map]*n_items).reshape(n_items,n_items)
    cluster_column = cluster_inverse_map.reshape(n_items,1)
    C = cluster_matrix == cluster_column

    category_matrix = np.array([category_inverse_map]*n_items).reshape(n_items,n_items)
    category_column = category_inverse_map.reshape(n_items,1)
    L = category_matrix == category_column
    #
    #   Step 3 - compute precision and recall
    #
    correctness = np.logical_and(C,L)  # where L(e) == L(e') iff C(e) == C(e')
    #
    #   compute precision
    #
    p_e_prime_ave = np.sum(correctness, axis=1)/np.sum(C != 0, axis=1)  # compute precision e' averages for each e where C is non-zero
    p = np.average(p_e_prime_ave)                                       # precision is the average of all e' averages
    #
    #   compute recall
    #
    r_e_prime_ave = np.sum(correctness, axis=1)/np.sum(L != 0, axis=1)  # compute recall e' averages for each e where L is non-zero
    r = np.average(r_e_prime_ave)                                       # recall is the average of all e' averages

    return p, r

def b_cubed(clusters, categories):
    '''
        Compute the extended b-cubed metric

        clusters  : a list of item sets representing computed item clusters
        categories: a list of item sets representing the given, true item categories

        item labels are assumed to a set of contiguous integers in the range [0..N-1] where N
        is the number of items

        return: precision, recall

        The algorithm is taken from "A comparison of extrinsic clustering evaluation metrics based on
        formal constraints", Amigo E, Gonsalo J, Artiles J, et al. May 11, 2009, https://www.researchgate.net/publication/225548032

    '''
    #
    #   Step 1 - compute inverse maps
    #
    max_item = -1
    for cluster in clusters:
        for item in cluster:
            max_item = max(max_item, item)
    n_items = max_item+1

    cluster_inverse_map = [set() for _ in range(n_items)]
    for i, cluster in enumerate(clusters):
        for item in cluster:
            cluster_inverse_map[item].add(i)

    category_inverse_map = [set() for _ in range(n_items)]
    for i, category in enumerate(categories):
        for item in category:
            category_inverse_map[item].add(i)

    #
    #   Step 2 - create intersection_count matrices
    #
    C = np.zeros((n_items,n_items), dtype=np.int32)
    L = np.zeros((n_items,n_items), dtype=np.int32)
    #
    #   populate the triangular matrices with the size of set intersections between sets
    #
    for e1 in range(n_items):
        for e2 in range(e1,n_items):
            C[e1,e2] = len(cluster_inverse_map[e1] & cluster_inverse_map[e2])
            L[e1,e2] = len(category_inverse_map[e1] & category_inverse_map[e2])
    #
    #   make the matrices symmetric since the intersection of (e,e') is the same as (e',e)
    #
    C = np.bitwise_or(C, C.T)
    L = np.bitwise_or(L, L.T)

    #
    #   Step 3 - compute precision and recall
    #
    #   We can safely ignore divide by zero exceptions and warnings since we replace those results with zeros
    #
    with np.errstate(divide='ignore', invalid='ignore'):
        c_l_min = np.minimum.reduce([C,L])  # find minimum between category and cluster intersections
        #
        #   compute precision
        #
        p = np.nan_to_num(c_l_min/C)                                # compute multi-precision for (e,e') pairs   
        p_e_prime_ave = np.sum(p, axis=1)/np.sum(C != 0, axis=1)    # compute multi-precision e' averages for each e where C is non-zero
        p = np.average(p_e_prime_ave)                               # precision is the average of all e' averages
        #
        #   compute recall
        #
        r = np.nan_to_num(c_l_min/L)                                # compute multi-recall for (e,e') pairs
        r_e_prime_ave = np.sum(r, axis=1)/np.sum(L != 0, axis=1)    # compute multi-recall e' averages for each e where L is non-zero
        r = np.average(r_e_prime_ave)                               # recall is the average of all e' averages

        return p, r


if __name__ == '__main__':
    import sys
    #
    #   example of using b_cubed with adjacency matrix cliques
    #
    #
    #   23-item reference (true) adjacency matrix
    #
    A_true = np.array([
        [0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0]
    ], dtype=np.int32)

    #
    #   23-item computed (inferred) adjacency matrix
    #
    A_inferred = np.array([
        [0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
        [0,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,0],
        [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0]
    ], dtype=np.int32)

    g_true = nx.Graph(A_true)
    cliques_true = list(nx.find_cliques(g_true))

    g_inferred = nx.Graph(A_inferred)
    cliques_inferred = list(nx.find_cliques(g_inferred))
    
    p,r = b_cubed(clusters=cliques_inferred, categories=cliques_true)
    f = f_score(p,r)
    
    print(f'precision: {p:5.4f}')
    print(f'recall   : {r:5.4f}')
    print(f'f_scores : {f:5.4f}')

class BCubedTest(unittest.TestCase):
    #
    #   parameters for 8 simple tests from examples in paper
    #       in these tests, an item may only have one label
    #
    simple_tests = [
        # Test 1a
        {
            'clusters'  :[[0,1,2,3],[4,5,6],[7,8,9,10,11,12,13]],
            'categories':[[0,1,2,3,4],[5,6,7,8,9,10],[11],[12],[13]],
            'p_expected':0.59863945,
            'r_expected':0.69523809,
            'f_expected':0.64333283},
        # Test 1b
        {
            'clusters'  : [[0,1,2,3],[4],[5,6],[7,8,9,10,11,12,13]],
            'categories': [[0,1,2,3,4],[5,6,7,8,9,10],[11],[12],[13]],
            'p_expected': 0.69387755,
            'r_expected': 0.69523809,
            'f_expected': 0.69455715},
        # Test 2a
        {
            'clusters'  : [[0,1,2,3],[4],[5,6],[7,8,9,10,11,12,13]],
            'categories': [[0,1,2,3,4,5,6],[7,8,9,10],[11],[12],[13]],
            'p_expected': 0.69387755,
            'r_expected': 0.71428571,
            'f_expected': 0.70393374},
        # Test 2b
        {
            'clusters'  : [[0,1,2,3],[4,5,6],[7,8,9,10,11,12,13]],
            'categories': [[0,1,2,3,4,5,6],[7,8,9,10],[11],[12],[13]],
            'p_expected': 0.69387755,
            'r_expected': 0.75510204,
            'f_expected': 0.72319632},
        # Test 3a
        {
            'clusters'  : [[0,1,2,3,4],[5,6,7,8]],
            'categories': [[0,1,2,3],[4],[5],[6],[7],[8]],
            'p_expected': 0.48888888,
            'r_expected': 1.00000000,
            'f_expected': 0.65671641},
        # Test 3b
        {
            'clusters'  : [[0,1,2,3],[4,5,6,7,8]],
            'categories': [[0,1,2,3],[4],[5],[6],[7],[8]],
            'p_expected': 0.55555555,
            'r_expected': 1.00000000,
            'f_expected': 0.71428571},
        # Test 4a
        {
            'clusters'  : [[0,1,2,3,4],[5],[6],[7],[8],[9],[10],[11],[12]],
            'categories': [[0,1,2,3,4],[5,6],[7,8],[9,10],[11,12]],
            'p_expected': 1.00000000,
            'r_expected': 0.69230769,
            'f_expected': 0.81818181},
        # Test 4b
        {
            'clusters'  : [[0,1,2,3],[4],[5,6],[7,8],[9,10],[11,12]],
            'categories': [[0,1,2,3,4],[5,6],[7,8],[9,10],[11,12]],
            'p_expected': 1.00000000,
            'r_expected': 0.87692307,
            'f_expected': 0.93442622}
    ]
    #
    #   parameters for 6 multiplicity tests from examples in paper
    #       in these tests, an item may have more than 1 label
    #
    multi_tests = [
        # Test 1
        {
            'clusters'  :[[0,1,2],[0,1,3,4],[5,6]],
            'categories':[[0,1,2],[0,1,3,4],[5,6]],
            'p_expected':1.00000000,
            'r_expected':1.00000000,
            'f_expected':1.00000000},
        # Test 2
        {
            'clusters'  : [[0,1,2],[0,1,2],[0,1,3,4],[5,6]],
            'categories': [[0,1,2],[0,1,3,4],[5,6]],
            'p_expected': 0.86190476,
            'r_expected': 1.00000000,
            'f_expected': 0.92583120},
        # Test 3
        {
            'clusters'  : [[0,1,2],[0,1,2],[0,1,2],[0,1,3,4],[5,6]],
            'categories': [[0,1,2],[0,1,3,4],[5,6]],
            'p_expected': 0.80952381,
            'r_expected': 1.00000000,
            'f_expected': 0.89473684},
        # Test4
        {
            'clusters'  : [[0,1,2],[3,4],[5,6]],
            'categories': [[0,1,2],[0,1,3,4],[5,6]],
            'p_expected': 1.00000000,
            'r_expected': 0.68571428,
            'f_expected': 0.81355932},
        # Test 5
        {
            'clusters'  : [[0,1,2],[0,1],[3,4],[5,6]],
            'categories': [[0,1,2],[0,1,3,4],[5,6]],
            'p_expected': 1.00000000,
            'r_expected': 0.74285714,
            'f_expected': 0.85245901},
        # Test 6
        {
            'clusters'  : [[0,1,2,3,4],[5,6]],
            'categories': [[0,1,2],[0,1,3,4],[5,6]],
            'p_expected': 0.88571428,
            'r_expected': 0.94285714,
            'f_expected': 0.91339285}
    ]

    # def test_common(self, clusters, categories, p_expected, r_expected, f_expected):
    def test_multi(self):
        for test in self.__class__.multi_tests:
            p,r = b_cubed(clusters=test['clusters'], categories=test['categories'])
            f = f_score(p,r)
            self.assertAlmostEqual(p, test['p_expected'], places=7)
            self.assertAlmostEqual(r, test['r_expected'], places=7)
            self.assertAlmostEqual(f, test['f_expected'], places=7)

    def test_simple(self):
        for test in self.__class__.simple_tests:
            p,r = b_cubed_simple(clusters=test['clusters'], categories=test['categories'])
            f = f_score(p,r)
            self.assertAlmostEqual(p, test['p_expected'], places=7)
            self.assertAlmostEqual(r, test['r_expected'], places=7)
            self.assertAlmostEqual(f, test['f_expected'], places=7)

            p,r = b_cubed(clusters=test['clusters'], categories=test['categories'])
            f = f_score(p,r)
            self.assertAlmostEqual(p, test['p_expected'], places=7)
            self.assertAlmostEqual(r, test['r_expected'], places=7)
            self.assertAlmostEqual(f, test['f_expected'], places=7)
