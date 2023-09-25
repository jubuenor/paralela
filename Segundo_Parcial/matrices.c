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
    int n_threads;
    double **matrixA ;
    double **matrixB ;
    
    // Thread
    pthread_t tid;
    
    // File with the values of the matrix
    FILE *file; 
    
    // Seconds for the Algorithm execution
    time_t seconds ;

    // maximum size of matrix
    n = 128 ;// 128,256,512,1024
    n_threads = 1;
    ////  Creación y lectura de las matrices ////
    
    // Asignación de memoria
    matrixA = (double **)malloc(n * sizeof(double *));
    matrixB = (double **)malloc(n * sizeof(double *));

    for (int i = 0; i < n; ++i) {
        matrixA[i] = (double *)malloc(n * sizeof(double));
        matrixB[i] = (double *)malloc(n * sizeof(double));
    }

    // Abrir archivo en modo binario
    file = fopen("matrix.txt", "r");
    if (file == NULL) {
        printf("No se pudo abrir el archivo.\n");
        exit(1);
    }

    // Lectura en bloque
    for (int i = 0; i < n; ++i) {
        fscanf(file,"%d",matrixA[i][i]);
    }

    fclose(file);

    // Algoritmo multiplicacion matriz

    // Creación de hilos
    // Let us create three threads
    /*
    
    for (int i  = 0; i < ; i++)
        pthread_create(&tid, NULL, multMatrix, (void *)&tid);
*/

    printf("%.2f ", matrixA[0][12]); 

    // Liberar memoria
    
    for (int i = 0; i < n; ++i) {
        free(matrixA[i]);
        free(matrixB[i]);
    }

    free(matrixA);
    free(matrixB);

    return 0;



}
