import struct
import random

def generate_random_matrix(n):
    return [[random.randrange(0, 100) for _ in range(n)] for _ in range(n)]

def write_matrix_to_binary_file(matrix, filename):
    with open(filename, 'wb') as f:
        for row in matrix:
            for item in row:
                # Empaquetar cada número de punto flotante en un formato binario
                packed_data = struct.pack('d', item)
                f.write(packed_data)

if __name__ == "__main__":
    n = 128  # Tamaño de la matriz
    matrix = generate_random_matrix(n)
    print(matrix)
    write_matrix_to_binary_file(matrix, 'matrix.bin')