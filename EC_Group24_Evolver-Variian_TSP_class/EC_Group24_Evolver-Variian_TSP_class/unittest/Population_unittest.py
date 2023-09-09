import random
from TSP import TSP
from Individual import Individual

class Population:
    def __init__(self, size, tsp_instance):
        self.individuals = [self.random_individual(
            tsp_instance) for _ in range(size)]
        self.tsp_instance = tsp_instance

    def random_individual(self, tsp_instance):
        # Creating a permutation in O(n) time complexity
        permutation = list(range(tsp_instance.num_cities))
        random.shuffle(permutation)
        return Individual(permutation, tsp_instance)

#unit testing the Population class
#the result of this test depends on the random seed, in this instance, the outcome is 
# different at each run, which indicates that the population is indeed random
if __name__ == '__main__':
    tsp_instance = TSP('unittest_sample.tsp')
    population = Population(2, tsp_instance)
    print(population.individuals[0].fitness)
    print(population.individuals[1].fitness)

