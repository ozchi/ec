import random
from TSP import TSP
from Individual import Individual

# random.seed(42)

class Population:
    def __init__(self, size, tsp_instance):
        self.population_size = size
        self.individuals = [self.random_individual(tsp_instance) for _ in range(size)]
        self.tsp_instance = tsp_instance

    def random_individual(self, tsp_instance):
        # Creating a permutation in O(n) time complexity
        permutation = list(range(tsp_instance.num_cities))
        random.shuffle(permutation)
        return Individual(permutation, tsp_instance)

    def fitness_proportional_selection(self):
        total_fitness = sum(ind.fitness for ind in self.individuals)
        # Avoid division by zero
        if total_fitness == 0:
            return random.choice(self.individuals)

        wheel_position = random.uniform(0, total_fitness)
        current_position = 0

        for individual in self.individuals:
            current_position += individual.fitness
            if current_position > wheel_position:
                return individual
        return self.individuals[-1]  # To handle potential floating-point errors


#unit testing the Population class
#the result of this test depends on the random seed, in this instance, the outcome is 
# different at each run, which indicates that the population is indeed random

from collections import Counter
from collections import defaultdict

def fitness_proportional_selection(population, num_selections):
    total_fitness = sum(individual.fitness for individual in population.individuals)
    selections = Counter()
    fitnesses = {}

    for _ in range(num_selections):
        selection_point = random.uniform(0, total_fitness)
        current_sum = 0
        for individual in population.individuals:
            current_sum += individual.fitness
            if current_sum >= selection_point:
                permutation = tuple(individual.permutation)
                selections[permutation] += 1
                fitnesses[permutation] = individual.fitness
                break

    return selections, fitnesses

def tournament_selection(population, num_selections, k=3):
    selections = defaultdict(int)
    fitnesses = {}

    for _ in range(num_selections):
        # Select k individuals randomly from the population
        tournament_contestants = random.sample(population.individuals, k)

        # Find the best individual among the selected contestants
        winner = max(tournament_contestants, key=lambda individual: individual.fitness)

        permutation = tuple(winner.permutation)
        selections[permutation] += 1
        fitnesses[permutation] = winner.fitness

    return selections, fitnesses

if __name__ == '__main__':
    tsp_instance = TSP('pcb442.tsp')
    population = Population(10, tsp_instance)

    num_selections = 100000
    selection_counts, selected_fitnesses = tournament_selection(population, num_selections)

    for permutation, count in selection_counts.items():
        fitness = selected_fitnesses[permutation]
        print(f"selected {count} times with fitness {int(fitness)}.")

