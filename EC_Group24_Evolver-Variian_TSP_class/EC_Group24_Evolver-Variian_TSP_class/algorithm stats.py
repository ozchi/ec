import matplotlib.pyplot as plt
import os
import csv
import numpy as np

def extract_final_fitness_from_file(filename):
    final_best_fitness = None

    with open(filename, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header row

        for row in csv_reader:
            final_best_fitness = int(row[3])  # Only care about the best fitness, and overwrite on every row.

    return final_best_fitness

if __name__ == "__main__":
    all_best_fitnesses = {20: [], 50: [], 100: []}

    for rep in range(30):  # for 30 repetitions
        for popsize in [20, 50, 100]:  # for each population size
            filename = f"EC_Group24_Evolver-Variian_TSP_class/EC_Group24_Evolver-Variian_TSP_class/out/results_rep_{rep}_popsize_{popsize}.txt"
            if os.path.exists(filename):
                final_fitness = extract_final_fitness_from_file(filename)
                all_best_fitnesses[popsize].append(final_fitness)

    # Plotting the box plot
    fig, ax = plt.subplots()
    box_plots = ax.boxplot(all_best_fitnesses.values())

    # Annotate the box plots with the statistics values
    for i, (popsize, values) in enumerate(all_best_fitnesses.items()):
        # i+1 because box_plots indices are 1-based
        whisker_lo = box_plots['whiskers'][2*i].get_ydata()[1]
        whisker_hi = box_plots['whiskers'][2*i+1].get_ydata()[1]
        q1 = np.percentile(values, 25)
        medi = np.median(values)
        q3 = np.percentile(values, 75)
        ax.text(i+1, whisker_lo, str(whisker_lo), va='center', ha='center', fontsize=8, color='red')
        ax.text(i+1, q1, str(q1), va='center', ha='center', fontsize=8, color='red')
        ax.text(i+1, medi, str(medi), va='center', ha='center', fontsize=8, color='red')
        ax.text(i+1, q3, str(q3), va='center', ha='center', fontsize=8, color='red')
        ax.text(i+1, whisker_hi, str(whisker_hi), va='center', ha='center', fontsize=8, color='red')

    ax.set_xticklabels(['20', '50', '100'])
    plt.xlabel('Population Size')
    plt.ylabel('Final Best Fitness')
    plt.title('Final Best Fitnesses across Repetitions for Different Population Sizes')
    plt.savefig('EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/algorithm_stats.png')
    plt.show()

