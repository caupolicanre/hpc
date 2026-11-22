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

    int p0 = 0;
    int p1 = 1;
    if (argc >= 3) {
        p0 = atoi(argv[1]);
        p1 = atoi(argv[2]);
    }

    const int num_sizes = 10;
    const int sizes[num_sizes] = {
        1 << 0,
        1 << 10,
        1 << 14,
        1 << 20,
        1 << 24,
        1 << 27,
        1 << 28,
        1 << 29,
        1 << 30
    };
    const int iters = 20;

    char *buf = NULL;
    if (rank == p0 || rank == p1) {
        buf = new char[sizes[num_sizes - 1]];
    }

    double avg_times[num_sizes];

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        double total_t = 0.0;

        // Iteracion -1 como warmup
        for (int k = -1; k < iters; k++) {
            double t0 = MPI_Wtime();

            if (rank == p0) {
                MPI_Send(buf, n, MPI_CHAR, p1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, n, MPI_CHAR, p1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == p1) {
                MPI_Recv(buf, n, MPI_CHAR, p0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, n, MPI_CHAR, p0, 0, MPI_COMM_WORLD);
            }

            double t1 = MPI_Wtime();
            if (k >= 0 && rank == p0) {
                total_t += (t1 - t0) / 2.0;
            }
        }

        if (rank == p0) {
            avg_times[s] = total_t / iters;
            printf("Bytes: %d | T_prom: %.9f s\n", n, avg_times[s]);
        }
    }

    if (rank == p0) {
        mkdir("results", 0777);
        FILE *fp = fopen("results/ejercicio1_datos.csv", "w");
        if (fp) {
            fprintf(fp, "bytes,tiempo_s\n");
            for (int s = 0; s < num_sizes; s++) {
                fprintf(fp, "%d,%.9f\n", sizes[s], avg_times[s]);
            }
            fclose(fp);
        }
    }

    if (rank == p0 || rank == p1) {
        delete[] buf;
    }

    MPI_Finalize();
    return 0;
}
