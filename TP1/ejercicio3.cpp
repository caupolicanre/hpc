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

    if (size < 2 || size % 2 != 0) {
        if (rank == 0) {
            fprintf(stderr, "Se requiere un numero par de procesos (n >= 2).\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int msg_bytes = 16 * 1024 * 1024; // 16 MB
    const int iters = 10;
    int mitad = size / 2;

    char *buf = new char[msg_bytes];
    double suma_max_t = 0.0;

    for (int k = -1; k < iters; k++) {
        MPI_Barrier(MPI_COMM_WORLD);

        double t0 = MPI_Wtime();
        if (rank < mitad) {
            MPI_Send(buf, msg_bytes, MPI_CHAR, rank + mitad, 0, MPI_COMM_WORLD);
        } else {
            MPI_Recv(buf, msg_bytes, MPI_CHAR, rank - mitad, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        double t1 = MPI_Wtime();
        double dt_local = t1 - t0;

        double max_t = dt_local;
        if (rank == 0) {
            for (int p = 1; p < size; p++) {
                double dt_p = 0.0;
                MPI_Recv(&dt_p, 1, MPI_DOUBLE, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if (dt_p > max_t) {
                    max_t = dt_p;
                }
            }
        } else {
            MPI_Send(&dt_local, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }

        if (k >= 0 && rank == 0) {
            suma_max_t += max_t;
        }
    }

    if (rank == 0) {
        double t_prom = suma_max_t / iters;
        double total_bytes = (double)msg_bytes * mitad;
        double bw_mbs = (total_bytes / (1024.0 * 1024.0)) / t_prom;
        double bw_mbps = (total_bytes * 8.0 / 1e6) / t_prom;

        printf("Procesos (n): %d | Pares: %d | BW diseccion: %.2f MB/s (%.2f Mbps)\n",
               size, mitad, bw_mbs, bw_mbps);

        mkdir("results", 0777);

        // Si el archivo no existe, escribo la cabecera; luego agrego la fila de esta corrida
        FILE *check = fopen("results/ejercicio3_diseccion.csv", "r");
        int existe = (check != NULL);
        if (check) fclose(check);

        FILE *fp = fopen("results/ejercicio3_diseccion.csv", "a");
        if (fp) {
            if (!existe) {
                fprintf(fp, "n_procesos,pares,tiempo_s,bw_MBs,bw_Mbps\n");
            }
            fprintf(fp, "%d,%d,%.6f,%.2f,%.2f\n", size, mitad, t_prom, bw_mbs, bw_mbps);
            fclose(fp);
        }
    }

    delete[] buf;
    MPI_Finalize();
    return 0;
}
