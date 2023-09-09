import cProfile
import random
from TSP import TSP
from Individual import Individual
from Population import Population
import matplotlib.pyplot as plt

# MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.2

# random.seed(42)

tsp_instance = TSP('pcb442.tsp')

def select_parents(population):
    parents = []
    for _ in range(2):
        parents.append(population.tournament_selection())
    return parents

def crossover(parents, tsp_instance):
    i = 0
    if random.random() < CROSSOVER_RATE:
        parent1 = parents[i]
        parent2 = parents[i + 1]
        child = Individual.order_crossover(parent1, parent2, tsp_instance)
        offspring = child
    else:
        offspring = parents[i]
    return offspring


def mutate(offspring, mutation_rate):

    child = offspring

    # Define a list of mutation functions
    mutation_functions = ["inversion_mutate",
                          "swap_mutate", "scramble_mutate", "insert_mutate"]

    choice = random.choice(mutation_functions)

    if random.random() < mutation_rate:
        if choice == "inversion_mutate":
            child = offspring.inversion_mutate(tsp_instance)
        elif choice == "swap_mutate":
            child = offspring.swap_mutate(tsp_instance)
        elif choice == "scramble_mutate":
            child = offspring.scramble_mutate(tsp_instance)
        elif choice == "insert_mutate":
            child = offspring.insert_mutate(tsp_instance)

    return child

def select_next_generation(population, offspring_list, tsp_instance, elitism_ratio=0.2):
    # Combine the population and offspring
    combined_population = population.individuals + offspring_list

    # Sort the combined population by fitness
    combined_population.sort(key=lambda ind: ind.fitness)

    # Determine the number of elite individuals to select based on the elitism ratio
    num_elite = int(elitism_ratio * population.population_size)

    # Select the top (elite) individuals
    selected_individuals = combined_population[:num_elite]

    # Select random individuals from the rest of the population to maintain diversity
    selected_individuals += random.sample(
        combined_population[num_elite:], population.population_size - num_elite)

    population.individuals = selected_individuals

def get_best_individual(population):
    best_fitness = float('inf')  # Initialize with a large value
    best_individual = None

    for ind in population.individuals:
        if ind.fitness < best_fitness:  # Assuming lower fitness is better
            best_fitness = ind.fitness
            best_individual = ind
    return best_individual

def evolutionary_algorithm(tsp_instance, population_sizes=[20], repetitions=1, max_generations=20000):

    results = []

    for rep in range(repetitions):

        for size in population_sizes:
            mutation_rate = 0.8
            mutation_rate_decay = (1/max_generations)/2
            filename = f'out/results_rep_{rep}_popsize_{size}.txt'
            
            # Write header to the file
            with open(filename, 'w') as file:
                file.write("Population Size, Generation Count, Repetition, Best Fitness, Median Fitness\n")
            
            population = Population(size, tsp_instance)

            best_fitness_values = []
            median_fitness_values = []

            for generation in range(max_generations):
                mutation_rate = mutation_rate - mutation_rate_decay
                offspring_list = []
                for i in range(population.population_size):
                    if generation % 10 == 0 and i == 1:
                        fitness_values = sorted([ind.fitness for ind in population.individuals])
                        best_fitness_values.append(fitness_values[0])
                        current_median_fitness = fitness_values[len(fitness_values) // 2]
                        median_fitness_values.append(fitness_values[len(fitness_values) // 2])
                        results.append((size, generation, rep, int(best_fitness_values[-1]), int(median_fitness_values[-1])))

                        print(
                            f"Generation Count: {generation}\tBest Fitness: {int(best_fitness_values[-1])}\tMedian Fitness: {int(median_fitness_values[-1])}\tPopulation Size: {size}\tRepetition: {rep}")
                                            # Append new results to the file
                        with open(filename, 'a') as file:
                            for size, generation, rep, best_fit, median_fit in results:
                                file.write(f"{size}, {generation}, {rep}, {best_fit}, {median_fit}\n")
                        results = []  # Clear the results for the next generation run
                                            # Check for changes in median fitness

                    parents = select_parents(population)
                    offspring = crossover(parents, tsp_instance)
                    offspring = mutate(offspring, mutation_rate)
                    offspring_list.append(offspring)

                select_next_generation(population, offspring_list, tsp_instance)

        best_individual = get_best_individual(population)
            
    return results, best_individual

if __name__ == '__main__':

    results, best_individual = evolutionary_algorithm(tsp_instance)

    print(best_individual.permutation)
