#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>

/*
- Compartidas (shared):
    Punteros/arreglos A, B, C y la dimension N. A y B son leidas concurrentemente por todos
    los hilos sin conflicto, mientras que cada hilo escribe en filas disjuntas de C (C[i*N + j]).
- Privadas (private):
    Indices de iteracion i, k, j y el escalar temporal a_ik. Deben ser privados para que cada
    hilo recorra su propio subconjunto de filas e indices sin pisar los contadores de los demas.
*/


void mat_mult_omp(const double *A, const double *B, double *C, int N, int num_threads)
{
    omp_set_num_threads(num_threads);

    #pragma omp parallel for default(none) shared(A, B, C, N) schedule(static)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0.0;
        }
        for (int k = 0; k < N; k++) {
            double a_ik = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

int main()
{
    const int n_dims = 4;
    const int dimensiones[n_dims] = {256, 512, 1024, 2048};
    const int n_hilos = 5;
    const int hilos[n_hilos] = {1, 2, 4, 8, 16};

    mkdir("results", 0777);
    FILE *fp = fopen("results/ejercicio3_matrices_omp.csv", "w");
    if (fp) {
        fprintf(fp, "N,threads,tiempo_s,speedup,eficiencia_pct,verificacion\n");
    }

    for (int d = 0; d < n_dims; d++) {
        int N = dimensiones[d];
        int total_elem = N * N;

        double *A = new double[total_elem];
        double *B = new double[total_elem];
        double *C = new double[total_elem];

        for (int idx = 0; idx < total_elem; idx++) {
            A[idx] = 1.0;
            B[idx] = 2.0;
            C[idx] = 0.0;
        }

        double t_seq = 0.0;

        for (int k = 0; k < n_hilos; k++) {
            int th = hilos[k];
            int reps = (N <= 512) ? 3 : 1;
            double acum_t = 0.0;

            for (int r = 0; r < reps; r++) {
                double t0 = omp_get_wtime();
                mat_mult_omp(A, B, C, N, th);
                double t1 = omp_get_wtime();
                acum_t += (t1 - t0);
            }
            double t_prom = acum_t / reps;

            if (th == 1) {
                t_seq = t_prom;
            }

            double speedup = t_seq / t_prom;
            double eficiencia = (speedup / th) * 100.0;

            printf("Matriz: %4dx%-4d | Threads: %2d | Tiempo: %.6f s | Speedup: %.4fx | Eficiencia: %.2f%% | C[0]: %.1f\n",
                   N, N, th, t_prom, speedup, eficiencia, C[0]);

            if (fp) {
                fprintf(fp, "%d,%d,%.6f,%.4f,%.2f,%.1f\n",
                        N, th, t_prom, speedup, eficiencia, C[0]);
            }
        }

        delete[] A;
        delete[] B;
        delete[] C;
    }

    if (fp) fclose(fp);
    return 0;
}
