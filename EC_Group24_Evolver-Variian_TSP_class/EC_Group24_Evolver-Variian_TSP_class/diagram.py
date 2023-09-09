import matplotlib.pyplot as plt


def read_tsp(filename):
    coords = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 3 and parts[0].isdigit():
                coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return coords

def plot_path(cities_coords, path, title, save_name):
    for city, (x, y) in cities_coords.items():
        plt.scatter(x, y, color='red', s=10)

    # Connecting adjacent cities in the path
    for i in range(1, len(path)):
        x1, y1 = cities_coords[path[i-1]]
        x2, y2 = cities_coords[path[i]]
        plt.plot([x1, x2], [y1, y2], color='blue')

    # Connecting the last city to the first one to complete the cycle
    x1, y1 = cities_coords[path[-1]]
    x2, y2 = cities_coords[path[0]]
    plt.plot([x1, x2], [y1, y2], color='blue')

    not_connected_cities = set(cities_coords.keys())
    for i in range(1, len(path)):
        not_connected_cities.discard(path[i-1])
        not_connected_cities.discard(path[i])

    print(f"{title} - not_connected:", not_connected_cities)

    missing_in_path = [city for city in not_connected_cities if city not in path]
    print(f"{title} - missing_in_p:", missing_in_path)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.savefig(save_name)
    plt.show()


cities_coords = read_tsp("EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class\pcb442.tsp")


best_path = [226, 411, 410, 237, 414, 420, 268, 416, 264, 263, 236, 218, 208, 197, 195, 196, 182, 169, 158, 146, 133, 134, 
147, 135, 159, 171, 184, 210, 199, 220, 209, 219, 198, 183, 170, 160, 148, 124, 101, 93, 62, 436, 112, 123, 111, 91, 92, 
90, 58, 59, 60, 61, 94, 125, 113, 383, 384, 98, 97, 96, 380, 379, 95, 64, 65, 32, 33, 377, 376, 31, 63, 30, 29, 28, 27, 26, 
25, 89, 57, 24, 23, 22, 21, 86, 52, 54, 55, 56, 88, 378, 110, 385, 109, 121, 122, 157, 145, 132, 391, 131, 120, 143, 144, 168, 
181, 156, 167, 180, 395, 394, 390, 388, 108, 381, 382, 87, 53, 85, 84, 83, 20, 19, 18, 439, 50, 82, 51, 17, 16, 49, 80, 15, 14, 
13, 12, 45, 79, 81, 100, 48, 47, 46, 78, 99, 77, 76, 44, 73, 74, 75, 42, 9, 10, 11, 43, 41, 72, 40, 8, 7, 6, 39, 5, 38, 71, 70, 
69, 68, 36, 37, 4, 3, 2, 67, 35, 442, 1, 34, 66, 102, 103, 114, 441, 104, 387, 116, 138, 392, 393, 175, 187, 211, 403, 399, 239, 
266, 271, 270, 267, 240, 241, 242, 407, 396, 174, 150, 162, 173, 186, 185, 172, 149, 126, 136, 161, 151, 115, 137, 386, 127, 389, 
152, 221, 212, 244, 243, 229, 222, 214, 402, 201, 202, 224, 232, 255, 233, 409, 404, 179, 178, 398, 192, 193, 194, 205, 207, 206, 
217, 204, 216, 225, 408, 413, 412, 261, 262, 438, 306, 345, 369, 370, 372, 371, 432, 427, 337, 336, 335, 373, 374, 375, 338, 307, 
437, 265, 275, 422, 419, 260, 259, 258, 257, 256, 254, 253, 252, 223, 231, 251, 250, 249, 415, 417, 418, 278, 297, 296, 294, 295, 
320, 354, 318, 290, 291, 292, 293, 319, 321, 430, 435, 359, 360, 361, 344, 325, 326, 327, 301, 429, 362, 431, 363, 364, 365, 366, 
367, 368, 334, 333, 328, 329, 330, 331, 332, 423, 272, 305, 304, 303, 302, 300, 299, 298, 324, 323, 322, 434, 355, 356, 357, 358, 
317, 316, 314, 343, 351, 352, 353, 313, 287, 312, 340, 350, 433, 349, 348, 347, 346, 339, 309, 310, 284, 283, 440, 308, 342, 341, 
428, 282, 277, 426, 274, 235, 228, 406, 227, 400, 401, 405, 234, 238, 273, 279, 276, 269, 281, 280, 285, 311, 286, 424, 421, 425, 
289, 315, 288, 247, 245, 246, 248, 230, 213, 200, 188, 163, 176, 153, 139, 164, 154, 191, 203, 215, 190, 165, 397, 189, 177, 166, 
155, 142, 130, 119, 141, 129, 118, 107, 140, 128, 117, 105, 106  
    
]

path_2 = [234, 227, 405, 400, 185, 172, 161, 149, 136, 126, 137, 127, 386, 441, 114, 103, 102, 66, 34, 442, 1, 2, 3, 36, 35, 67, 68, 
37, 4, 5, 6, 39, 72, 71, 38, 70, 69, 104, 115, 387, 389, 152, 151, 150, 162, 173, 186, 174, 396, 399, 187, 403, 407, 401, 406, 228, 235, 
239, 238, 266, 269, 273, 276, 279, 281, 282, 277, 274, 271, 270, 267, 240, 241, 242, 243, 244, 245, 246, 247, 229, 221, 212, 211, 175, 392, 
138, 116, 99, 77, 44, 76, 75, 74, 73, 41, 40, 7, 8, 9, 42, 43, 10, 11, 12, 45, 46, 78, 79, 100, 80, 48, 47, 13, 14, 15, 16, 17, 50, 49, 81, 
82, 439, 84, 83, 51, 18, 19, 52, 20, 53, 85, 381, 382, 86, 87, 378, 88, 89, 90, 91, 92, 61, 62, 63, 95, 96, 380, 379, 94, 436, 101, 93, 60, 
59, 58, 57, 56, 55, 54, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 64, 31, 32, 376, 377, 33, 65, 97, 98, 384, 383, 113, 125, 135, 134, 124, 112, 
111, 123, 133, 146, 158, 169, 182, 197, 196, 195, 181, 168, 157, 145, 144, 391, 132, 122, 110, 121, 385, 109, 120, 388, 131, 143, 156, 394, 
167, 180, 192, 193, 204, 216, 225, 233, 408, 409, 404, 217, 205, 194, 206, 207, 208, 218, 410, 411, 219, 209, 198, 183, 170, 159, 147, 148, 
160, 171, 184, 199, 210, 220, 226, 414, 237, 265, 437, 275, 423, 420, 268, 416, 264, 272, 438, 422, 419, 260, 261, 262, 263, 236, 413, 412, 
259, 258, 257, 256, 255, 254, 253, 418, 417, 278, 298, 297, 296, 322, 323, 430, 324, 429, 325, 299, 300, 301, 326, 327, 302, 303, 304, 305, 
306, 331, 332, 333, 334, 307, 335, 336, 427, 337, 338, 375, 374, 373, 372, 371, 432, 370, 369, 368, 367, 366, 365, 345, 330, 329, 328, 431, 
364, 363, 362, 344, 361, 360, 359, 435, 358, 357, 356, 434, 355, 354, 353, 343, 352, 351, 350, 349, 433, 348, 347, 346, 342, 341, 428, 308, 
309, 339, 310, 283, 440, 426, 280, 284, 285, 311, 286, 287, 312, 340, 313, 314, 315, 316, 317, 318, 319, 320, 321, 295, 294, 293, 292, 291, 
290, 289, 288, 424, 421, 425, 248, 249, 415, 250, 251, 252, 232, 224, 215, 203, 190, 191, 398, 178, 179, 395, 166, 155, 142, 390, 130, 119, 
108, 107, 118, 129, 141, 154, 164, 165, 177, 397, 189, 202, 201, 402, 214, 223, 231, 230, 222, 213, 200, 188, 176, 163, 393, 153, 139, 140, 
128, 117, 105, 106
    ]

path_3 = [405, 406, 234, 266, 239, 240, 242, 243, 245, 421, 425, 424, 288, 287, 312, 340, 350, 351, 352, 353, 343, 434, 355, 354, 316, 290, 
289, 291, 292, 317, 318, 314, 315, 313, 286, 284, 311, 285, 310, 309, 308, 283, 280, 440, 426, 277, 274, 270, 269, 273, 276, 279, 341, 342, 
346, 348, 347, 349, 433, 339, 428, 281, 282, 271, 267, 238, 227, 400, 401, 185, 172, 136, 115, 386, 441, 127, 387, 389, 137, 150, 162, 173, 
396, 174, 151, 392, 152, 138, 393, 175, 211, 403, 221, 229, 212, 213, 222, 231, 223, 214, 215, 190, 203, 191, 398, 179, 166, 178, 395, 390,
 388, 120, 109, 121, 145, 391, 132, 122, 110, 385, 131, 144, 143, 156, 155, 142, 164, 165, 189, 201, 202, 402, 200, 188, 176, 397, 177, 167, 
 394, 180, 157, 168, 181, 194, 195, 207, 196, 182, 197, 208, 218, 410, 411, 414, 237, 265, 437, 338, 337, 333, 332, 330, 304, 303, 302, 327, 
 301, 326, 329, 305, 438, 272, 420, 268, 423, 306, 331, 345, 367, 366, 365, 368, 369, 370, 432, 371, 372, 373, 374, 375, 336, 427, 335, 334, 
 307, 275, 416, 264, 206, 205, 409, 217, 408, 412, 413, 261, 260, 259, 258, 257, 418, 417, 251, 250, 293, 319, 320, 294, 295, 278, 299, 298, 297, 323, 324, 356, 357, 
358, 359, 430, 344, 362, 363, 364, 361, 360, 435, 296, 321, 322, 429, 300, 325, 431, 328, 422, 419, 262, 263, 236, 404, 193, 192, 204, 216, 
233, 225, 256, 255, 224, 232, 254, 253, 252, 415, 249, 230, 248, 247, 246, 244, 241, 235, 228, 407, 187, 399, 186, 161, 149, 126, 114, 103, 
102, 68, 69, 70, 116, 104, 66, 67, 35, 442, 34, 1, 2, 36, 37, 38, 6, 7, 3, 4, 5, 39, 71, 72, 42, 74, 75, 73, 40, 41, 8, 9, 10, 43, 76, 44, 
11, 77, 45, 14, 13, 12, 46, 47, 15, 16, 49, 48, 79, 78, 99, 105, 117, 106, 118, 130, 119, 108, 129, 154, 141, 140, 139, 153, 163, 128, 107, 
439, 100, 80, 81, 82, 51, 83, 50, 17, 18, 19, 20, 52, 84, 53, 54, 21, 22, 55, 56, 88, 87, 378, 382, 381, 85, 86, 23, 24, 25, 58, 57, 89, 90, 
28, 29, 30, 32, 33, 377, 376, 64, 31, 63, 62, 61, 93, 60, 27, 26, 59, 92, 91, 111, 133, 123, 101, 436, 95, 94, 96, 65, 98, 384, 383, 113, 97, 379, 380, 
112, 124, 134, 147, 199, 226, 220, 210, 184, 171, 125, 135, 148, 160, 159, 170, 219, 209, 198, 183, 169, 158, 146  
    
]


plot_path(cities_coords, best_path, "EA1", "EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/algorithm solutions1.png")
plot_path(cities_coords, path_2, "EA2", "EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/algorithm solutions2.png")
plot_path(cities_coords, path_3, "EA3","EC_Group24_Evolver-Variian_TSP_class\EC_Group24_Evolver-Variian_TSP_class/results/algorithm solutions3.png")

