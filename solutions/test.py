"""Sanity checks for codelab solutions."""
import numpy as np
import part1_sols as p1
import part3_sols as p3

# Test k-rank approximate.
approx = p1.k_rank_approximate(np.identity(4), 3)
print approx

# Test train_ls_classifier.
A = np.mat([[1, 2, 3, 4],
            [4, 5, 6, 7],
            [7, 8, 9, 11]])
y = np.mat([[1], [-1], [1]])
LS = p3.train_ls_classifier(A, y)
print LS