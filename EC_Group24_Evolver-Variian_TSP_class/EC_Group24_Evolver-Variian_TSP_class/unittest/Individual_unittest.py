from TSP import TSP

class Individual:
    def __init__(self, permutation, tsp_instance):
        self.permutation = permutation
        self.fitness = self.calculate_fitness(tsp_instance)

    def calculate_fitness(self, tsp_instance):
        total_distance = 0
        for i in range(len(self.permutation) - 1):
            city1 = self.permutation[i]
            city2 = self.permutation[i + 1]
            total_distance += tsp_instance.get_distance(city1, city2)
        return total_distance

    # Other methods like mutation, crossover, etc.

tsp_instance = TSP('unittest_sample.tsp')

individual = Individual([2, 0, 1], tsp_instance)

print(individual.fitness) # Prints the total distance of the tour