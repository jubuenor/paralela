#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

 
// maximum number of threads
#define MAX_THREAD // 1,2,4,8,16
 


// Cada fila y columna será unicamente procesada por un hilo, para al finalizar realizar la sumatoria respectiva

int main() {
    // Variables

    // Matrix
    int n ;
    double **matrixA ;
    double **matrixB ;
    
    // File with the values of the matrix
    FILE *file; 
    
    // Seconds for the Algorithm execution
    time_t seconds ;

    // maximum size of matrix
    n = 4 ;// 128,256,512,1024
    
    // Asignación de memoria
    matrixA = (double **)malloc(n * sizeof(double *));
    matrixB = (double **)malloc(n * sizeof(double *));

    for (int i = 0; i < n; ++i) {
        matrixA[i] = (double *)malloc(n * sizeof(double));
        matrixB[i] = (double *)malloc(n * sizeof(double));
    }

    // Abrir archivo en modo binario
    file = fopen("matrix.bin", "rb");
    if (file == NULL) {
        printf("No se pudo abrir el archivo.\n");
        exit(1);
    }

    // Lectura en bloque
    for (int i = 0; i < n; ++i) {
        fread(matrixA[i], sizeof(double), n, file);
        fread(matrixB[i], sizeof(double), n, file);
    }

    fclose(file);

    // Creación y lectura de las matrices

    // Algoritmo multiplicacion matriz


    printf("%.2f ", matrixA[0][0]); 

    // Liberar memoria
    
    for (int i = 0; i < n; ++i) {
        free(matrixA[i]);
        free(matrixB[i]);
    }

    free(matrixA);
    free(matrixB);

    return 0;



}
