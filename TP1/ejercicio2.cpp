#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>


int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Se requieren al menos 2 procesos.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int msg_bytes = 16 * 1024 * 1024; // 16 MB
    const int iters = 10;

    char *buf = new char[msg_bytes];
    double *fila_local = new double[size];
    for (int j = 0; j < size; j++) {
        fila_local[j] = 0.0;
    }

    for (int orig = 0; orig < size; orig++) {
        for (int dest = 0; dest < size; dest++) {
            if (orig == dest) continue;

            MPI_Barrier(MPI_COMM_WORLD);

            double suma_t = 0.0;
            for (int k = -1; k < iters; k++) {
                double t0 = MPI_Wtime();

                if (rank == orig) {
                    MPI_Send(buf, msg_bytes, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
                    MPI_Recv(buf, msg_bytes, MPI_CHAR, dest, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else if (rank == dest) {
                    MPI_Recv(buf, msg_bytes, MPI_CHAR, orig, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Send(buf, msg_bytes, MPI_CHAR, orig, 0, MPI_COMM_WORLD);
                }

                double t1 = MPI_Wtime();
                if (k >= 0 && rank == orig) {
                    suma_t += (t1 - t0) / 2.0;
                }
            }

            if (rank == orig) {
                double t_prom = suma_t / iters;
                double bw_mbs = (msg_bytes / (1024.0 * 1024.0)) / t_prom;
                fila_local[dest] = bw_mbs;
            }
        }
    }

    double *matriz = NULL;
    if (rank == 0) {
        matriz = new double[size * size];
        for (int j = 0; j < size; j++) {
            matriz[j] = fila_local[j];
        }
        for (int p = 1; p < size; p++) {
            MPI_Recv(&matriz[p * size], size, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    } else {
        MPI_Send(fila_local, size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Matriz de transferencia (MB/s):\n");
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                printf("%10.2f ", matriz[i * size + j]);
            }
            printf("\n");
        }

        mkdir("results", 0777);
        FILE *fp = fopen("results/ejercicio2_matriz.csv", "w");
        if (fp) {
            fprintf(fp, "origen");
            for (int j = 0; j < size; j++) {
                fprintf(fp, ",P%d", j);
            }
            fprintf(fp, "\n");
            for (int i = 0; i < size; i++) {
                fprintf(fp, "P%d", i);
                for (int j = 0; j < size; j++) {
                    fprintf(fp, ",%.2f", matriz[i * size + j]);
                }
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
        delete[] matriz;
    }

    delete[] fila_local;
    delete[] buf;

    MPI_Finalize();
    return 0;
}
