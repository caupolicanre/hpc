#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>


bool es_primo(int n)
{
    if (n <= 1) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    int limite = (int)sqrt((double)n);
    for (int d = 3; d <= limite; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 5) {
        if (rank == 0) {
            fprintf(stderr, "Este ejercicio requiere exactamente 5 procesos.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int limite_max = 10000000;
    const int n_part = 5;
    const int particiones[n_part] = {1000, 10000, 100000, 1000000, 2000000};

    double *tiempos_calc = NULL;
    double *tiempos_sync = NULL;
    int *conteos_locales = NULL;

    if (rank == 0) {
        tiempos_calc = new double[size];
        tiempos_sync = new double[size];
        conteos_locales = new int[size];

        mkdir("results", 0777);
        FILE *fp = fopen("results/ejercicio3_primos.csv", "w");
        if (fp) {
            fprintf(fp, "particion,proceso,t_calculo_s,t_sync_s,t_total_s,primos_locales,primos_totales\n");
            fclose(fp);
        }
    }

    for (int p = 0; p < n_part; p++) {
        int tam_bloque = particiones[p];
        int primos_loc = 0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0_calc = MPI_Wtime();

        for (int base = 0; base < limite_max; base += size * tam_bloque) {
            int ini = base + rank * tam_bloque + 1;
            int fin = ini + tam_bloque - 1;

            if (ini > limite_max) break;
            if (fin > limite_max) fin = limite_max;

            for (int num = ini; num <= fin; num++) {
                if (es_primo(num)) {
                    primos_loc++;
                }
            }
        }

        double t1_calc = MPI_Wtime();
        double dt_calc = t1_calc - t0_calc;

        double t0_sync = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        int primos_glob = 0;
        MPI_Reduce(&primos_loc, &primos_glob, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        double t1_sync = MPI_Wtime();
        double dt_sync = t1_sync - t0_sync;

        MPI_Gather(&dt_calc, 1, MPI_DOUBLE, tiempos_calc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&dt_sync, 1, MPI_DOUBLE, tiempos_sync, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&primos_loc, 1, MPI_INT, conteos_locales, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Particion: %d | Primos totales: %d\n", tam_bloque, primos_glob);
            FILE *fp = fopen("results/ejercicio3_primos.csv", "a");

            for (int i = 0; i < size; i++) {
                double t_tot = tiempos_calc[i] + tiempos_sync[i];
                printf("  P%d -> Calculo: %.6f s | Sync/Comm: %.6f s | Total: %.6f s | Primos: %d\n",
                       i, tiempos_calc[i], tiempos_sync[i], t_tot, conteos_locales[i]);
                if (fp) {
                    fprintf(fp, "%d,%d,%.6f,%.6f,%.6f,%d,%d\n",
                            tam_bloque, i, tiempos_calc[i], tiempos_sync[i], t_tot, conteos_locales[i], primos_glob);
                }
            }
            if (fp) fclose(fp);
        }
    }

    if (rank == 0) {
        delete[] tiempos_calc;
        delete[] tiempos_sync;
        delete[] conteos_locales;
    }

    MPI_Finalize();
    return 0;
}
