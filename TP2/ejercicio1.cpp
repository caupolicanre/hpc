#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>


int mybcast_secuencial(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == root) {
        for (int dst = 0; dst < size; dst++) {
            if (dst != root) {
                MPI_Send(buffer, count, datatype, dst, 0, comm);
            }
        }
    } else {
        MPI_Recv(buffer, count, datatype, root, 0, comm, MPI_STATUS_IGNORE);
    }

    return MPI_SUCCESS;
}

int mybcast_arbol(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int vrank = (rank - root + size) % size;

    int paso = 1;
    while (paso < size) {
        paso <<= 1;
    }
    paso >>= 1;

    for (; paso > 0; paso >>= 1) {
        int bloque = paso << 1;

        if (vrank % bloque == 0) {
            int vdst = vrank + paso;
            if (vdst < size) {
                int dst = (vdst + root) % size;
                MPI_Send(buffer, count, datatype, dst, 0, comm);
            }
        } else if (vrank % bloque == paso) {
            int vsrc = vrank - paso;
            int src = (vsrc + root) % size;
            MPI_Recv(buffer, count, datatype, src, 0, comm, MPI_STATUS_IGNORE);
        }
    }

    return MPI_SUCCESS;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int n_bytes = 16 * 1024 * 1024; // 16 MB
    const int iters = 10;

    char *buf = new char[n_bytes];
    if (rank == 0) {
        for (int i = 0; i < n_bytes; i++) {
            buf[i] = 10;
        }
    }

    // Warmup
    mybcast_secuencial(buf, n_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);
    mybcast_arbol(buf, n_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double suma_seq = 0.0;
    for (int k = 0; k < iters; k++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        mybcast_secuencial(buf, n_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        double dt_local = t1 - t0;
        double dt_max = 0.0;
        MPI_Reduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            suma_seq += dt_max;
        }
    }

    double suma_arb = 0.0;
    for (int k = 0; k < iters; k++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        mybcast_arbol(buf, n_bytes, MPI_CHAR, 0, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        double dt_local = t1 - t0;
        double dt_max = 0.0;
        MPI_Reduce(&dt_local, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            suma_arb += dt_max;
        }
    }

    if (rank == 0) {
        double prom_seq_ms = (suma_seq / iters) * 1000.0;
        double prom_arb_ms = (suma_arb / iters) * 1000.0;

        printf("Procesos: %d | T_secuencial: %.4f ms | T_arbol: %.4f ms\n",
               size, prom_seq_ms, prom_arb_ms);

        mkdir("results", 0777);
        FILE *check = fopen("results/ejercicio1_bcast.csv", "r");
        int existe = (check != NULL);
        if (check) fclose(check);

        FILE *fp = fopen("results/ejercicio1_bcast.csv", "a");
        if (fp) {
            if (!existe) {
                fprintf(fp, "procesos,t_secuencial_ms,t_arbol_ms\n");
            }
            fprintf(fp, "%d,%.4f,%.4f\n", size, prom_seq_ms, prom_arb_ms);
            fclose(fp);
        }
    }

    delete[] buf;
    MPI_Finalize();
    return 0;
}
