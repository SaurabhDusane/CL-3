import numpy as np

# fuzzy operations
def fuzzy_union(A, B):
    return np.maximum(A, B)

def fuzzy_intersection(A, B):
    return np.minimum(A, B)

def fuzzy_complement(A):
    return 1 - A

def fuzzy_difference(A, B):
    return np.maximum(A - B, 0)

def cartesian_product(A, B):
    return np.outer(A, B)

def max_min_composition(R1, R2):
    return np.fmax.outer(R1, R2)

# fuzzy arrays
A = np.array([0.2, 0.5, 0.8])
B = np.array([0.1, 0.4, 0.6])

# perform fuzzy set operation
union_result = fuzzy_union(A, B)
intersection_result = fuzzy_intersection(A, B)
complement_result_A = fuzzy_complement(A)
difference_result = fuzzy_difference(A, B)

# print the operations o/p
print("Union:", union_result)
print("Intersection:", intersection_result)
print("Complement of A:", complement_result_A)
print("Difference A - B:", difference_result)

# Create fuzzy relations
R1 = np.array([0.2, 0.4, 0.6])
R2 = np.array([0.3, 0.7, 0.9])

# Perform Cartesian product
cartesian_result = cartesian_product(R1, R2) # different values for the cartesian product
print("Cartesian product result:")
print(cartesian_result)

# Create fuzzy relations
R1 = np.array([[0.2, 0.4, 0.6],
               [0.3, 0.7, 0.9]])

R2 = np.array([[0.1, 0.5],
               [0.6, 0.2],
               [0.7, 0.3]])

# Perform max-min composition
composition_result = max_min_composition(R1, R2) # for max-min operation we take a different value
print("Max-min composition result:")
print(composition_result)
