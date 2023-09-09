import time
import random
from TSP import TSP
from Individual import Individual
from Population import Population
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure 'results' directory exists
if not os.path.exists('results'):
    os.makedirs('results')

random.seed(42)

# Initialize the evolutionary algorithm
class EvolutionaryAlgorithm:
    def __init__(self, tsp_instance, population_size, crossover_rate, elitism_rate):
        self.tsp_instance = tsp_instance            
        self.population_size = population_size      
        self.crossover_rate = crossover_rate        
        self.elitism_rate = elitism_rate            
        self.population = Population(population_size, tsp_instance)
        self.best_individual_ever = None  
        self.repetition = 1      

    def mutate(self, individual, mutation_rate):
        mutate_strategies = [individual.swap_mutate, 
                    individual.inversion_mutate, 
                    individual.insert_mutate, 
                    individual.scramble_mutate]
        chosen_strategy = random.choice(mutate_strategies)
        mutated_individual = chosen_strategy(self.tsp_instance)
        
        if random.random() < mutation_rate:
            mutated_individual = self.mutate(mutated_individual, mutation_rate)
        return mutated_individual

    def run(self, generations):
        best_fitnesses = []
        median_fitnesses = []
        
        MR = 0.05

        for generation in range(generations):
            mutation_rate = MR
            offspring = []
            
            num_elites = int(self.elitism_rate * self.population_size)
            
            self.population.individuals.sort(key=lambda ind: ind.fitness)
            
            elites = self.population.individuals[:num_elites]
            offspring.extend(elites)

            while len(offspring) < self.population_size:
                parent1 = random.choice(self.population.individuals)
                parent2 = random.choice(self.population.individuals)

                if random.random() < self.crossover_rate:
                    child = Individual.order_crossover(parent1, parent2, self.tsp_instance)
                else:
                    child = Individual(parent1.permutation.copy(), self.tsp_instance)  

                child = self.mutate(child, mutation_rate)
                offspring.append(child)

            for ind in offspring:
                ind.calculate_fitness(self.tsp_instance)

            combined_population = self.population.individuals + offspring
            combined_population.sort(key=lambda ind: ind.fitness)
            selected_individuals = combined_population[:self.population_size]
            
            self.population.individuals = selected_individuals
            
            # Update the best individual
            if self.best_individual_ever is None or self.best_individual_ever.fitness > self.population.individuals[0].fitness:
                self.best_individual_ever = self.population.individuals[0]

            if generation % 10 == 0:
                fitness_values = [ind.fitness for ind in self.population.individuals]
                best_fitness = min(fitness_values)
                median_fitness = np.median(fitness_values)
                
                best_fitnesses.append(best_fitness)
                median_fitnesses.append(median_fitness)
            
            print(f"Gen {generation} || Best Fit: {best_fitness:.7f} || Median Fit: {median_fitness:.7f} || MR: {mutation_rate:.3f} || Repetition: {self.repetition}")
        
            if generation == 20000:
                self.repetition += 1
                
        return best_fitnesses, median_fitnesses, self.best_individual_ever  # Added returning the best individual ever

if __name__ == '__main__':
    tsp_file_path = 'EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class\pcb442.tsp'
    tsp_data = TSP(tsp_file_path)
    
    start_time = time.time()
    generations = 20000
    crossover_rate = 0.4    
    elitism_rate = 0.1

    best_global_individual = None

    results = {}

    for population_size in [20, 50, 100]:
        for _ in range(30):
            pop = Population(population_size, tsp_data)
            ea = EvolutionaryAlgorithm(tsp_data, population_size, crossover_rate, elitism_rate)
            best_fitnesses, median_fitnesses, best_individual = ea.run(generations)
            
            if population_size not in results:
                results[population_size] = []
            results[population_size].append((best_fitnesses, median_fitnesses))
            
            # Update the global best individual if this one is better
            if best_global_individual is None or best_individual.fitness < best_global_individual.fitness:
                best_global_individual = best_individual

    best_path = best_global_individual.permutation
    
    for city in best_path:
        if city >= tsp_data.num_cities:  
            print(f"City index {city} is out of range!")
            best_path.remove(city)

    with open(f"EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/best_path_global.txt", "w") as f:
        f.write(" ,".join(map(str, best_path)))

   
    for i in range(30):
        plt.figure()
        
        for population_size in [20, 50, 100]:
            best_fitnesses, median_fitnesses = results[population_size][i]
            plt.plot(best_fitnesses, label=f"Best Fitness {population_size}")
            plt.plot(median_fitnesses, label=f"Median Fitness {population_size}")

        plt.xlabel('Generation (every 10th)')
        plt.ylabel('Fitness')
        plt.title(f'Fitness over Generations - Repetition {i+1}')
        plt.legend()
        plt.savefig(f'EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/repetition{i+1}.png')
        plt.close()

    print("--- %s seconds ---" % (time.time() - start_time))
    
    with open(f'EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/fitness_data_1.txt', 'w') as f:
        for population_size, repetitions in results.items():
            for i, (best_fitnesses, median_fitnesses) in enumerate(repetitions):
                f.write(f"Population Size: {population_size}, Repetition: {i + 1}\n")
                f.write("Best Fitnesses:\n")
                f.write(",".join(map(str, best_fitnesses)) + "\n")
                f.write("Median Fitnesses:\n")
                f.write(",".join(map(str, median_fitnesses)) + "\n")
                f.write("\n")

 
    '''
    with open(f'results/fitness_data_2.txt', 'w') as f:
        for population_size, repetitions in results.items():
            for i, (best_fitnesses, median_fitnesses) in enumerate(repetitions):
                f.write(f"Population Size: {population_size}, Repetition: {repetitions},Best Fitness: {str(best_fitnesses)}\n"
                   f"Median Fitnesses: {median_fitnesses}\n")
    '''
  

    elapsed_time = time.time() - start_time
    print(f"Time elapsed: {elapsed_time:.2f} seconds.")