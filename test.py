"""Sanity checks for codelab solutions."""
import numpy as np
import toolbox as tb

# Test k-rank approximate.
approx = tb.k_rank_approximate(np.identity(4), 3)
print approx

# Test train_ls_classifier.
A = np.mat([[1, 2, 3, 4],
            [4, 5, 6, 7],
            [7, 8, 9, 11]])
y = np.mat([[1], [-1], [1]])
LS = tb.train_ls_classifier(A, y)

print LS