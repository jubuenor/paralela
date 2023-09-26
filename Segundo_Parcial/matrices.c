#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MAX_THREAD 1 // Number of threads
#define SIZE 128     // Matrix size

int step_i = 0;
pthread_mutex_t lock;

// Function to print the matrix
void printMatrix(double **matrix, int size)
{
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void *multi(void *arg)
{
    pthread_mutex_lock(&lock);
    int i = step_i++;
    pthread_mutex_unlock(&lock);

    double **matrixA = ((double ***)arg)[0];
    double **matrixB = ((double ***)arg)[1];
    double **matrixC = ((double ***)arg)[2];

    for (int i = 0; i < SIZE; i++)
    {
        for (int j = 0; j < SIZE; j++)
        {
            for (int k = 0; k < SIZE; k++)
            {
                matrixC[i][j] += matrixA[i][k] * matrixB[k][j];
            }
        }
    }

    return NULL;
}

int main()
{
    double **matrixA, **matrixB, **matrixC;
    pthread_t threads[MAX_THREAD];
    pthread_mutex_init(&lock, NULL);
    FILE *file;

    // Memory allocation
    matrixA = (double **)malloc(SIZE * sizeof(double *));
    matrixB = (double **)malloc(SIZE * sizeof(double *));
    matrixC = (double **)malloc(SIZE * sizeof(double *));
    for (int i = 0; i < SIZE; ++i)
    {
        matrixA[i] = (double *)malloc(SIZE * sizeof(double));
        matrixB[i] = (double *)malloc(SIZE * sizeof(double));
        matrixC[i] = (double *)malloc(SIZE * sizeof(double));
    }

    // Initialize matrices (for demonstration)
    // Abrir archivo en modo binario
    file = fopen("/home/user/Desktop/MateoCodes/nacional/Paralela/paralela/Segundo_Parcial/matrix.txt", "r");

    if (file == NULL)
    {
        printf("No se pudo abrir el archivo.\n");
        exit(1);
    }
    int value;

    // Reading the matrix
    for (int i = 0; i < SIZE; ++i)
    {

        for (int j = 0; j < SIZE; ++j)
        {
            fscanf(file, "%d", &value);

            matrixA[i][j] = value;
            matrixB[i][j] = value;
        }
    }

    double **arg[3] = {matrixA, matrixB, matrixC};

    // Thread creation
    for (int i = 0; i < MAX_THREAD; i++)
    {
        pthread_create(&threads[i], NULL, multi, arg);
    }

    // Join threads
    for (int i = 0; i < MAX_THREAD; i++)
    {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&lock);

    // Print the resultant matrix
    printf("Resultant Matrix C:\n");
    printMatrix(matrixC, SIZE);

    // Free memory
    for (int i = 0; i < SIZE; ++i)
    {
        free(matrixA[i]);
        free(matrixB[i]);
        free(matrixC[i]);
    }
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
