import random
from TSP import TSP
from Individual import Individual


class Population:
    def __init__(self, size, tsp_instance):
        self.population_size = size
        self.individuals = [self.random_individual(tsp_instance) for _ in range(size)]
        self.tsp_instance = tsp_instance

    @classmethod
    def from_individuals(cls, individuals, tsp_instance):
        instance = cls(len(individuals), tsp_instance)
        instance.individuals = individuals
        return instance

    def random_individual(self, tsp_instance):
        # Creating a permutation in O(n) time complexity
        permutation = list(range(tsp_instance.num_cities))
        random.shuffle(permutation)
        return Individual(permutation, tsp_instance)

    def fitness_proportional_selection(self):
        # Invert fitness values so that lower fitness has higher selection probability
        inverted_fitnesses = [max(individual.fitness for individual in self.individuals) + 1 - individual.fitness for individual in self.individuals]
        total_fitness = sum(inverted_fitnesses)

        # Avoid division by zero
        if total_fitness == 0:
            return random.choice(self.individuals)

        wheel_position = random.uniform(0, total_fitness)
        current_position = 0

        for i, individual in enumerate(self.individuals):
            current_position += inverted_fitnesses[i]
            if current_position > wheel_position:
                return individual

        return self.individuals[-1]

    def tournament_selection(self, tournament_group_size = 4):
        # Select a random group of individuals for the tournament
        competitors = random.sample(self.individuals, tournament_group_size)
        # Sort them based on their fitness
        competitors.sort(key=lambda x: x.fitness)
        # Return the best individual from the tournament
        return competitors[0]

# unit testing the Population class
# the result of this test depends on the random seed, in this instance, the outcome is
# different at each run, which indicates that the population is indeed random
if __name__ == '__main__':
    tsp_instance = TSP('pcb442.tsp')
    population = Population(10, tsp_instance)

    num_selections = 100
    k = 3
    selected_individuals = Population.tournament_selection(
        population, num_selections, k)

    # Printing selected individuals
    for i, individual in enumerate(selected_individuals):
        print(
            f"Selection {i + 1}: Permutation {individual.permutation}, Fitness {individual.fitness}")
