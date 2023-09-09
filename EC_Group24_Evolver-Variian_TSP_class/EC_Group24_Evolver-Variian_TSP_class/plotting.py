import matplotlib.pyplot as plt
import os
import csv

def plot_from_file(filename):
    generation_counts = []
    best_fitnesses = []
    median_fitnesses = []

    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header row

        for row in csv_reader:
            generation_count = int(row[1])
            best_fitness = int(row[3])
            median_fitness = int(row[4])

            generation_counts.append(generation_count)
            best_fitnesses.append(best_fitness)
            median_fitnesses.append(median_fitness)

    plt.plot(generation_counts, best_fitnesses, label=f"Best Fitness ({filename.split('_')[-1][:-4]})")
    plt.plot(generation_counts, median_fitnesses, label=f"Median Fitness ({filename.split('_')[-1][:-4]})")


if __name__ == "__main__":
    # Assuming all files are in the same directory as the script and their names follow the pattern 'results_rep_x_popsize_y.txt'
    for rep in range(30):  # for 30 repetitions
        for popsize in [20, 50, 100]:  # for each population size
            filename = f"out/results_rep_{rep}_popsize_{popsize}.txt"
            if os.path.exists(filename):
                plot_from_file(filename)

        plt.xlabel('Generation Count')
        plt.ylabel('Fitness')
        plt.title(f'Fitness over Generations (Repetition {rep})')
        plt.legend()
        # plt.show()
        plt.savefig(f'EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/EA1repetition{rep}.png')  # Save the plot
        plt.clf()  # Clear the current plot for the next one