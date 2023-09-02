#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>

#define NUM_HILOS 4
#define N_TERMS 5e4

long double sum = 0;

void *funcion(void *i)
{
    long long local_i = (long long)i;
    long double sum_local = 0;
    for (; local_i < (long long)i + N_TERMS; local_i++)
    {
        // printf("\n %lld", local_i & (long long)1);
        sum_local = sum_local + (local_i & 1 ? (long double)-1 / (2 * local_i + 1) : (long double)1 / (2 * local_i + 1));
        // sum_local += pow(-1, local_i) / (2 * local_i + 1);
    }
    int hilo = local_i / N_TERMS;
    sum += sum_local;
    printf("\n Hilo %d - %.15Lf", hilo, 4 * sum_local);
}

int main()
{

    pthread_t hilo[NUM_HILOS];
    int r, *rh0, i;
    for (i = 0; i < NUM_HILOS; i++)
    {
        long long TERM = i * N_TERMS;
        r = pthread_create(&hilo[i], NULL, (void *)funcion, (void *)TERM);
        if (r != 0)
        {
            perror("\n-->pthread_create error: ");
            exit(-1);
        }
    }

    for (i = 0; i < NUM_HILOS; i++)
    {
        r = pthread_join(hilo[i], (void **)&rh0);
        if (r != 0)
        {
            perror("\n-->pthread_join error: ");
            exit(-1);
        }
    }

    printf("\n Pi =  %.15Lf", 4 * sum);
    printf("\n");
    return 0;
}