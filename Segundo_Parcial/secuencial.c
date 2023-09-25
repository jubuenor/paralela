#include <stdio.h>
#include <stdlib.h>

#define SIZE 2 // Matrix size

// Function to print the matrix
void printMatrix(double **matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    double **matrixA, **matrixB, **matrixC;
    FILE *file;

    // Memory allocation
    matrixA = (double **)malloc(SIZE * sizeof(double *));
    matrixB = (double **)malloc(SIZE * sizeof(double *));
    matrixC = (double **)malloc(SIZE * sizeof(double *));
    for (int i = 0; i < SIZE; ++i) {
        matrixA[i] = (double *)malloc(SIZE * sizeof(double));
        matrixB[i] = (double *)malloc(SIZE * sizeof(double));
        matrixC[i] = (double *)malloc(SIZE * sizeof(double));
    }

    // Initialize matrices
    file = fopen("/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Segundo_Parcial/matrix.txt", "r");
    if (file == NULL) {
        printf("No se pudo abrir el archivo.\n");
        exit(1);
    }
    int value;

    // Reading the matrix
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            fscanf(file, "%d", &value);
            matrixA[i][j] = value;
            matrixB[i][j] = value;
        }
    }

    // Sequential matrix multiplication
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            matrixC[i][j] = 0;
            for (int k = 0; k < SIZE; k++) {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    // Print the resultant matrix
    printf("Resultant Matrix C:\n");
    printMatrix(matrixC, SIZE);

    // Free memory
    for (int i = 0; i < SIZE; ++i) {
        free(matrixA[i]);
        free(matrixB[i]);
        free(matrixC[i]);
    }
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
