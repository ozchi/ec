from TSP import TSP
import random

class Individual:
    id = 0

    def __init__(self, permutation, tsp_instance):
        #路径
        self.permutation = permutation
        #距离   
        self.fitness = self.calculate_fitness(tsp_instance)
        #含有城市数量
        self.num_cities = tsp_instance.num_cities
        #id
        self.id = Individual.id
        Individual.id += 1

    def calculate_fitness(self, tsp_instance):
        total_distance = 0
        for i in range(len(self.permutation) - 1):
            city1 = self.permutation[i]
            city2 = self.permutation[i + 1]
            total_distance += tsp_instance.get_distance(city1, city2)
        return total_distance

    # following methods create a new individual, which is a mutated version of the current individual
    # it does so by swapping two random cities in the permutation
    def swap_mutate(self, tsp_instance):
        mutated_permutation = self.permutation.copy()
        i, j = random.sample(range(len(self.permutation)),
                             2)  # Pick two random positions
        # Swap the cities
        mutated_permutation[i], mutated_permutation[j] = mutated_permutation[j], mutated_permutation[i]
        return Individual(mutated_permutation, tsp_instance)

    # it does so by inserting a random city in a random position in the permutation
    def insert_mutate(self, tsp_instance):
        # Create a copy of the permutation to mutate
        mutated_permutation = self.permutation.copy()

        # Pick two random positions
        i, j = random.sample(range(len(mutated_permutation)), 2)

        # Make sure i comes before j
        if i > j:
            i, j = j, i

        # Remove the city at position j and insert it right after position i
        city_to_move = mutated_permutation.pop(j)
        mutated_permutation.insert(i + 1, city_to_move)

        # Return a new Individual with the mutated permutation
        return Individual(mutated_permutation, tsp_instance)

    # it does so by inverting a random substring of the permutation
    def inversion_mutate(self, tsp_instance):
        # Create a copy of the permutation to mutate
        mutated_permutation = self.permutation.copy()

        # Pick two random positions
        i, j = random.sample(range(len(mutated_permutation)), 2)

        # Make sure i comes before j
        if i > j:
            i, j = j, i

        # Invert the substring between the two positions
        mutated_permutation[i:j+1] = reversed(mutated_permutation[i:j+1])

        # Return a new Individual with the mutated permutation
        return Individual(mutated_permutation, tsp_instance)

    # it does so by scrambling a random subset of the permutation
    def scramble_mutate(self, tsp_instance):
        # Create a copy of the permutation to mutate
        mutated_permutation = self.permutation.copy()

        # Determine the size of the subset to scramble (e.g., 1/4 of the total length)
        # generate a random number between 0 and len(mutated_permutation), but it has to be at least 1 and also an integer
        subset_size = max(1, int(random.random() * len(mutated_permutation)))

        # Pick a subset of random positions
        subset_indices = random.sample(
            range(len(mutated_permutation)), subset_size)

        # Extract the genes (cities) at those positions
        subset_values = [mutated_permutation[i] for i in subset_indices]

        # Shuffle the subset of values
        temp = random.shuffle(subset_values)

        # Place the shuffled values back into the permutation
        for index, value in zip(subset_indices, subset_values):
            mutated_permutation[index] = value

        # Return a new Individual with the mutated permutation
        return Individual(mutated_permutation, tsp_instance)

    # following methods create a new individual (child), which is a crossover of the current individual and another individual (parents)
    # and crossovers are static methods, because they don't need to access the current individual's attributes

    # it does so by copying a random subset from one parent, and filling in the remaining cities from the other parent
    @staticmethod
    def order_crossover(parent1, parent2, tsp_instance):
        # Determine crossover points
        start, end = sorted(random.sample(range(len(parent1.permutation)), 2))

        # Extract the subset from the first parent
        subset = parent1.permutation[start:end+1]

        # Initialize the child's permutation with None
        child_permutation = [None] * len(parent1.permutation)

        # Copy the subset into the child
        child_permutation[start:end+1] = subset

        # Fill in the remaining cities from the second parent
        pointer = 0
        for i in range(len(parent1.permutation)):
            if child_permutation[i] is None:
                while parent2.permutation[pointer] in subset:
                    pointer += 1
                child_permutation[i] = parent2.permutation[pointer]
                pointer += 1

        # Create and return a new Individual object with the child's permutation
        return Individual(child_permutation, tsp_instance)

        # Apply the mapping to the child's genes outside the crossover points

    # it does so by copying a random subset from one parent, and filling in the remaining cities from the other parent
    @staticmethod
    def pmx_crossover(parent1, parent2, tsp_instance):
        start, end = sorted(random.sample(range(len(parent1.permutation)), 2))

        # Create mappings for the genes between the crossover points
        mapping = {}
        for i in range(start, end):
            mapping[parent1.permutation[i]] = parent2.permutation[i]
            mapping[parent2.permutation[i]] = parent1.permutation[i]

        # Copy the segment from parent1 into the child
        child_permutation = [None] * len(parent1.permutation)
        for i in range(start, end):
            child_permutation[i] = parent1.permutation[i]

        visited_genes = set()
        for i in list(range(0, start)) + list(range(end, len(parent2.permutation))):
            gene = parent2.permutation[i]
            while gene in child_permutation and gene not in visited_genes:
                visited_genes.add(gene)
                gene = mapping.get(gene, gene)
            child_permutation[i] = gene

        return Individual(child_permutation, tsp_instance)

    # this implementation of cycle crossover is deterministic
    @staticmethod
    def cycle_crossover(parent1, parent2, tsp_instance):

        n = len(parent1.permutation)
        used_indices = set()
        child_permutation = [None] * n

        # Helper function to find a cycle
        def find_cycle(start):
            cycle = []
            i = start
            while True:
                cycle.append(i)
                gene = parent1.permutation[i]
                i = parent2.permutation.index(gene)
                if i == start:
                    break
            return cycle

        use_parent1 = True
        for i in range(n):
            if i not in used_indices:
                cycle = find_cycle(i)
                used_indices.update(cycle)
                for j in cycle:
                    child_permutation[j] = (
                        parent1 if use_parent1 else parent2).permutation[j]
                use_parent1 = not use_parent1

        return Individual(child_permutation, tsp_instance)

    @staticmethod
    def edge_recombination(parent1, parent2, tsp_instance):
        # Create the edge table
        edge_table = {i: set() for i in range(len(parent1.permutation))}
        for parent in [parent1, parent2]:
            for i in range(len(parent.permutation)):
                edge_table[parent.permutation[i]].add(
                    parent.permutation[(i - 1) % len(parent.permutation)])
                edge_table[parent.permutation[i]].add(
                    parent.permutation[(i + 1) % len(parent.permutation)])

        # Start with a random city
        current_city = random.choice(parent1.permutation)
        child_permutation = [current_city]

        # Build the child permutation
        while len(child_permutation) < len(parent1.permutation):
            # Remove current city from edge table
            for neighbors in edge_table.values():
                neighbors.discard(current_city)

            # Select the next city with the fewest neighbors
            next_city = min(edge_table[current_city], key=lambda x: len(
                edge_table[x]), default=None)

            # If there is no next city, choose a random unvisited city
            if next_city is None:
                unvisited_cities = set(
                    parent1.permutation) - set(child_permutation)
                next_city = random.choice(list(unvisited_cities))

            child_permutation.append(next_city)
            current_city = next_city

        return Individual(child_permutation, tsp_instance)

# testing the Individual class
if __name__ == '__main__':
    # random.seed(33)
    tsp_instance = TSP('unittest_sample.tsp')
    parent1 = Individual([1, 2, 3, 4, 6, 0, 5], tsp_instance)
    parent2 = Individual([0, 3, 4, 5, 6, 1, 2], tsp_instance)

    print(parent1.permutation)
    print(parent2.permutation)
    print("gap")
    child = parent1.edge_recombination(parent1, parent2, tsp_instance)
    print(child.permutation)
