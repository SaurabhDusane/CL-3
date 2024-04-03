import random
import numpy as np

# Define problem-specific parameters
num_dimensions = 5
population_size = 20
max_generations = 50
mutation_rate = 0.1

# Define the cost/fitness function (replace this with your problem-specific function)
def fitness_function(individual):
    return sum(individual)

# Initialize population with random values
population = [np.random.rand(num_dimensions) for _ in range(population_size)]

# Main optimization loop
for generation in range(max_generations):
    # Evaluate the fitness of each individual in the population
    fitness_values = [fitness_function(individual) for individual in population]

    # Perform cloning based on fitness values
    clones = []
    for i in range(population_size):
        num_clones = int(population_size / (fitness_values[i] + 0.0001))  # Ensure non-zero denominator
        clones.extend([population[i]] * num_clones)

    # Mutation step
    for i in range(len(clones)):
        for j in range(num_dimensions):
            if random.random() < mutation_rate:
                clones[i][j] = random.random()

    # Select top individuals for the next generation
    population = sorted(clones, key=lambda x: fitness_function(x))[:population_size]

# Get the best individual from the final population
best_individual = min(population, key=fitness_function)
best_fitness = fitness_function(best_individual)

# Print the best individual and its fitness value
print("Best Individual:", best_individual)
print("Best Fitness Value:", best_fitness)

# In this code:
# You can customize num_dimensions, population_size, max_generations, mutation_rate, and the fitness_function to suit your problem.
# The algorithm iterates through generations, evaluates fitness, clones individuals based on fitness, mutates them, and selects the best individuals for the next generation.
# Finally, it prints out the best individual and its fitness value after the optimization process.
