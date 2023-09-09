from math import sqrt

class TSP:
    def __init__(self, file_path):
        # distances is a 2D array, where distances[i][j] is the distance between city i and city j
        # num_cities is the number of cities
        self.distances, self.num_cities = self.load_distances(file_path)

    def get_distance(self, city1, city2):
        return self.distances[city1][city2]
    
    def load_distances(self, file_path):
        return load_distances_helper(file_path)

# This function loads the x,y coordnates of cities from a file, and computes the distances between them
# It returns a 2D array, where distances[i][j] is the distance between city i and city j
def load_distances_helper(file_path):
    cities = []
    with open(file_path, 'r') as file:
        # Skip the metadata lines
        # this implementation might be a bit fragile, because it depends on the file format
        # eg it has to end with "NODE_COORD_SECTION", and the cities' coordinates have to be in the following lines
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                break

        # Read the cities' coordinates
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # Check if the line contains three elements
                _, x, y = map(float, parts)
                cities.append((x, y))

    # Compute the distances between cities
    distances = [[0] * len(cities) for _ in range(len(cities))]
    for i in range(len(cities)):
        for j in range(i+1, len(cities)):
            dx = cities[i][0] - cities[j][0]
            dy = cities[i][1] - cities[j][1]
            distance = sqrt(dx * dx + dy * dy)
            distances[i][j] = distance
            distances[j][i] = distance # Since it's a symmetric TSP

    #distances is a 2D array, where distances[i][j] is the distance between city i and city j
    #len(cities) is the number of cities
    return distances, len(cities)

if __name__ == '__main__':
    tsp = TSP('pcb442.tsp')
    print(tsp.get_distance(0, 1))
    print(tsp.get_distance(1, 2))
    print(tsp.num_cities)