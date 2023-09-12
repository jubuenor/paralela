#include <stdio.h>
#include "omp.h"

#define N_TERMS 2e9

#define N_THREADS 4
long double init, end;
long double total_time;

void main()
{
    long double pi = 0;
    int i;
    init = omp_get_wtime();

    omp_set_num_threads(N_THREADS);
    printf("\n Hilos = %d ", N_THREADS);
#pragma omp parallel num_threads(N_THREADS)
#pragma omp for reduction(+ : pi)
    for (i = 0; i < (int)N_TERMS; i++)
    {

        pi += (i & 1 ? (long double)-1 / (2 * i + 1) : (long double)1 / (2 * i + 1));
    }
    end = omp_get_wtime();
    total_time = end - init;

    printf("\n Pi = %.15Lf \n Total time = %Lf\n", 4 * pi, total_time);
}