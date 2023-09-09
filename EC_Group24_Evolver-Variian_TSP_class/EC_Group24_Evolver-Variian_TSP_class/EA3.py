import random
from TSP import TSP
from Individual import Individual
from Population import Population
import matplotlib.pyplot as plt
import numpy as np
import os

if not os.path.exists('results'):
    os.makedirs('results')

def mutate(individual, tsp_instance, mutation_rate):
    mutation_choice = random.choice(['swap', 'insert', 'invert', 'scramble'])
    if mutation_choice == 'swap':
        individual = individual.swap_mutate(tsp_instance)
    elif mutation_choice == 'insert':
        individual = individual.insert_mutate(tsp_instance)
    elif mutation_choice == 'invert':
        individual = individual.inversion_mutate(tsp_instance)
    elif mutation_choice == 'scramble':
        individual = individual.scramble_mutate(tsp_instance)
    if random.random() < mutation_rate:
        individual = mutate(individual, tsp_instance, mutation_rate)
    return individual

def tournament_pick(pop, tournament_group_size):
    competitors = random.sample(pop.individuals, tournament_group_size)
    competitors.sort(key=lambda x: x.fitness)
    return competitors[0]

def execute_genetic_algorithm(tsp_data, pop, gens, cross_rate, cross_method):
    base_mutation_rate = 0.01 
    least_mutation_rate = 0.001 
    maximum_gens = 10000

    elitism_ratio = 0.1
    elite_count = int(elitism_ratio * pop_size)

    best_fitnesses = []
    median_fitnesses = []

    for gen in range(gens):
        mutation_prob = max(least_mutation_rate, base_mutation_rate * (1 - gen / maximum_gens))
        next_generation = []

        for _ in range(pop_size):
            group_size = 4
            parent_one = tournament_pick(pop, group_size)
            parent_two = tournament_pick(pop, group_size)
            if random.random() < cross_rate:
                if cross_method == 'order_crossover':
                    child_entity = Individual.order_crossover(parent_one, parent_two, tsp_data)
                elif cross_method == 'pmx_crossover':
                    child_entity = Individual.pmx_crossover(parent_one, parent_two, tsp_data)
                elif cross_method == 'cycle_crossover':
                    child_entity = Individual.cycle_crossover(parent_one, parent_two, tsp_data)
                elif cross_method == 'edge_recombination': 
                    child_entity = Individual.edge_recombination(parent_one, parent_two, tsp_data)
            else:
                child_entity = Individual(parent_one.permutation.copy(), tsp_data)
            child_entity = mutate(child_entity, tsp_data, mutation_prob)
            next_generation.append(child_entity)

        elites = sorted(pop.individuals, key=lambda x: x.fitness)[:elite_count]
        all_entities = elites + next_generation
        all_entities.sort(key=lambda x: x.fitness)
        pop.individuals = all_entities[:pop_size]

        if gen % 10 == 0:
            fit_scores = [entity.fitness for entity in pop.individuals]
            best_fitness = min(fit_scores)
            median_fitness = np.median(fit_scores)
            best_fitnesses.append(best_fitness)
            median_fitnesses.append(median_fitness)
        print(f"[Gen {gen}] Best Fitness: {best_fitness} | Median Fitness: {median_fitness} | Mutation Prob: {mutation_prob:.4f}")

    best_paths = [entity.permutation for entity in sorted(pop.individuals, key=lambda x: x.fitness)[:1]]
    return best_fitnesses, median_fitnesses, best_paths[-1]

if __name__ == '__main__':
    tsp_file_path = 'EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class\pcb442.tsp'
    tsp_data = TSP(tsp_file_path)
    
    cross_rate = 0.5 
    cross_method = 'order_crossover'
    gens = 20000

    results = {}

    for pop_size in [20, 50, 100]:
        results[pop_size] = []
        for repetition in range(30):
            pop = Population(pop_size, tsp_data)
            best_fitnesses, median_fitnesses, best_path = execute_genetic_algorithm(tsp_data, pop, gens, cross_rate, cross_method)
            
            # Save the best path for this iteration
        if repetition == 29:
            with open(f"results/best_path_pop_{pop_size}_run_{repetition+1}.txt", "w") as f:
                f.write(" ,".join(map(str, best_path)))

            results[pop_size].append((best_fitnesses, median_fitnesses))

    for i in range(30):
        plt.figure()
        for pop_size in [20, 50, 100]:
            best_fitnesses, median_fitnesses = results[pop_size][i]
            plt.plot(best_fitnesses, label=f"Best Fitness {pop_size}")
            plt.plot(median_fitnesses, label=f"Median Fitness {pop_size}")

        plt.xlabel('Generation (every 10th)')
        plt.ylabel('Fitness')
        plt.title(f'Fitness over Generations - Run {i+1}')
        plt.legend()
        plt.savefig(f'results/repetition{i+1}.png')
        plt.close()
