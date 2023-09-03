#include <stdio.h>
#include "omp.h"
#include <math.h>

#define N_THREADS 16
#define N_TERMS 2e9 / N_THREADS

long long i[N_THREADS];

long double pi[N_THREADS];

void main()
{
    for (int j = 0; j < N_THREADS; j++)
    {
        i[j] = N_TERMS * (j + 1);
    }
#pragma omp parallel num_threads(N_THREADS)
    {
        int id = omp_get_thread_num();
        long long j = i[id] - N_TERMS;
        for (; j < i[id]; j++)
        {
            pi[id] += (j & 1 ? (long double)-1 / (2 * j + 1) : (long double)1 / (2 * j + 1));

            // pi[id] += pow(-1, j) / (2 * j + 1);
        }
    }
    long double pi_total = 0;
    for (int j = 0; j < N_THREADS; j++)
    {
        pi_total += pi[j];
        printf("\n Hilo %d =  %.15Lf", j, 4 * pi[j]);
    }
    printf("\n Pi = %.15Lf \n", 4 * pi_total);
}