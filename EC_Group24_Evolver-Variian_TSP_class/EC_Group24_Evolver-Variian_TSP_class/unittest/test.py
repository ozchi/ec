import cProfile
import random
from TSP import TSP
from Individual import Individual
from Population import Population
import matplotlib.pyplot as plt

# MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.7

# random.seed(42)

tsp_instance = TSP('pcb442.tsp')


def select_parents(population):
    parents = []
    for _ in range(2):
        parents.append(population.tournament_selection())
    return parents


def crossover(parents, tsp_instance):
    for i in range(0, len(parents)-1):
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

    if random.random() < mutation_rate:
        # Randomly select a mutation function from the list
        if random.choice(mutation_functions) == "inversion_mutate":
            child = offspring.inversion_mutate(tsp_instance)
        elif random.choice(mutation_functions) == "swap_mutate":
            child = offspring.swap_mutate(tsp_instance)
        elif random.choice(mutation_functions) == "scramble_mutate":
            child = offspring.scramble_mutate(tsp_instance)
        elif random.choice(mutation_functions) == "insert_mutate":
            child = offspring.insert_mutate(tsp_instance)

    return child


def select_next_generation(population, offspring_list, tsp_instance, elitism_ratio=0.1):
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

    # Create a new population from the selected individuals
    population = Population.from_individuals(
        selected_individuals, tsp_instance)

    return population

def evolutionary_algorithm(tsp_instance, population_sizes=[20], repetitions=1, max_generations=2):
    mutation_rate = 0.8
    mutation_rate_decay = (1/max_generations)/2
    results = []

    for rep in range(repetitions):
        # plt.figure(rep)  # Create a figure for each repetition
        for size in population_sizes:
            population = Population(size, tsp_instance)

            best_fitness_values = []
            median_fitness_values = []

            for generation in range(max_generations):
                mutation_rate = mutation_rate - mutation_rate_decay
                # generate the same amount of offspring as the population size
                for i in range(population.population_size):
                    if (generation) % 1 == 0 and i == 0:
                        # print the average fitness of the population
                        fitness_values = [
                            ind.fitness for ind in population.individuals]
                        median_fitness = sorted(fitness_values)[
                            len(fitness_values) // 2]
                        best_fitness = min(fitness_values)
                        print(f"Generation: {generation}\tBest Fitness: {int(best_fitness)}\tMedian Fitness: {int(median_fitness)}\tPopulation Size: {size}\tRepetition: {rep}")
                    # 2. Selection
                    parents = select_parents(population)

                    # 3. Crossover
                    offspring = crossover(parents, tsp_instance)

                    # 4. Mutation
                    offspring = mutate(offspring, mutation_rate)

                    offspring_list = []
                    offspring_list.append(offspring)

                    # 5. Replacement
                    population = select_next_generation(
                        population, offspring_list, tsp_instance)
                    # Calculate average fitness
                    # Calculate and store the best and median fitness values
                    fitness_values = sorted(
                        [ind.fitness for ind in population.individuals])
                    
                    #extract the individual with the best fitness
                    best_individual = [ind for ind in population.individuals if ind.fitness == fitness_values[0]][0]

                    best_fitness_values.append(fitness_values[0])

                    median_fitness_values.append(fitness_values[len(fitness_values) // 2])

                    results.append((size, rep, best_fitness, median_fitness_values, best_individual))

        #     plt.plot(best_fitness_values,label=f'Best Fitness (Pop size {size})')
        #     plt.plot(median_fitness_values,label=f'Median Fitness (Pop size {size})')
        # plt.xlabel('Generation')
        # plt.ylabel('Fitness')
        # plt.title(f'Fitness over Generations (Repetition {rep+1})')
        # plt.legend()
        # plt.savefig(f'figures/fitness_over_generations_rep_{rep+1}.png')
        # plt.show()

    return results

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()  # Start profiling
    results = evolutionary_algorithm(tsp_instance)
    profiler.disable()  # Stop profiling
    profiler.dump_stats('profile_data.cprof')
    best_result = min(results, key=lambda x: x[2])  # x[2] corresponds to the best_fitness in each tuple
    size, rep, best_fitness,_, best_individual = best_result
    with open('best_individual.txt', 'w') as file:
        file.write(f"Population Size: {size}, Repetition: {rep},Best Fitness: {int(best_fitness)}\n"
                   f"Best Individual: {best_individual.permutation}\n")
