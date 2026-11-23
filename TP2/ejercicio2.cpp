#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>


void mat_mult(const double *local_A, const double *B, double *local_C, int N, int filas_locales)
{
    int total_elem = filas_locales * N;
    for (int idx = 0; idx < total_elem; idx++) {
        local_C[idx] = 0.0;
    }

    for (int i = 0; i < filas_locales; i++) {
        for (int k = 0; k < N; k++) {
            double a_ik = local_A[i * N + k];
            for (int j = 0; j < N; j++) {
                local_C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_pruebas = 3;
    const int dimensiones[n_pruebas] = {512, 1024, 2048};

    for (int t = 0; t < n_pruebas; t++) {
        int N = dimensiones[t];

        if (N % size != 0) {
            if (rank == 0) {
                fprintf(stderr, "La dimension N (%d) debe ser divisible por la cantidad de procesos (%d).\n", N, size);
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int filas_locales = N / size;
        int elem_locales = filas_locales * N;
        int elem_totales = N * N;

        double *A = NULL;
        double *C = NULL;
        double *B = new double[elem_totales];
        double *local_A = new double[elem_locales];
        double *local_C = new double[elem_locales];

        if (rank == 0) {
            A = new double[elem_totales];
            C = new double[elem_totales];
            for (int i = 0; i < elem_totales; i++) {
                A[i] = 1.0;
                B[i] = 2.0;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        MPI_Bcast(B, elem_totales, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatter(A, elem_locales, MPI_DOUBLE, local_A, elem_locales, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        mat_mult(local_A, B, local_C, N, filas_locales);

        MPI_Gather(local_C, elem_locales, MPI_DOUBLE, C, elem_locales, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double t1 = MPI_Wtime();
        double dt_local = t1 - t0;
        double dt_max = 0.0;

        MPI_Reduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Procesos: %d | Matriz: %dx%d | Tiempo: %.6f s | C[0]: %.1f (Esperado: %.1f)\n",
                   size, N, N, dt_max, C[0], N * 2.0);

            mkdir("results", 0777);
            FILE *check = fopen("results/ejercicio2_matrices.csv", "r");
            int existe = (check != NULL);
            if (check) fclose(check);

            FILE *fp = fopen("results/ejercicio2_matrices.csv", "a");
            if (fp) {
                if (!existe) {
                    fprintf(fp, "procesos,N,tiempo_s,verificacion\n");
                }
                fprintf(fp, "%d,%d,%.6f,%.1f\n", size, N, dt_max, C[0]);
                fclose(fp);
            }
        }

        delete[] B;
        delete[] local_A;
        delete[] local_C;
        if (rank == 0) {
            delete[] A;
            delete[] C;
        }
    }

    MPI_Finalize();
    return 0;
}
