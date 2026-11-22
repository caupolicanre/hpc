#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <mpi.h>


int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char *ruta = "homo_sapiens_chromosome_1.fasta";
    if (argc >= 2) {
        ruta = argv[1];
    }

    long long tam_bloque = 0;
    char *bloque = NULL;

    if (rank == 0) {
        FILE *fp = fopen(ruta, "rb");
        if (!fp) {
            fprintf(stderr, "Error al abrir %s\n", ruta);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Saltar primera linea (header FASTA)
        int ch;
        while ((ch = fgetc(fp)) != EOF && ch != '\n') {}

        long long inicio = ftell(fp);
        fseek(fp, 0, SEEK_END);
        long long fin = ftell(fp);
        long long total_datos = fin - inicio;
        fseek(fp, inicio, SEEK_SET);

        long long base = total_datos / size;
        long long resto = total_datos % size;

        tam_bloque = base + (0 < resto ? 1 : 0);
        bloque = new char[tam_bloque];
        size_t leidos = fread(bloque, 1, tam_bloque, fp);
        (void)leidos;

        for (int p = 1; p < size; p++) {
            long long p_tam = base + (p < resto ? 1 : 0);
            char *tmp = new char[p_tam];
            size_t p_leidos = fread(tmp, 1, p_tam, fp);
            (void)p_leidos;

            MPI_Send(&p_tam, 1, MPI_LONG_LONG, p, 0, MPI_COMM_WORLD);
            MPI_Send(tmp, (int)p_tam, MPI_CHAR, p, 1, MPI_COMM_WORLD);
            delete[] tmp;
        }
        fclose(fp);
    } else {
        MPI_Recv(&tam_bloque, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        bloque = new char[tam_bloque];
        MPI_Recv(bloque, (int)tam_bloque, MPI_CHAR, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    long long local_A = 0;
    long long local_validas = 0;

    for (long long i = 0; i < tam_bloque; i++) {
        unsigned char c = (unsigned char)bloque[i];
        if (!isalpha(c)) continue;

        char mayus = (char)toupper(c);
        if (mayus == 'A') {
            local_A++;
        }
        if (mayus != 'N') {
            local_validas++;
        }
    }

    if (rank == 0) {
        long long *arr_A = new long long[size];
        long long *arr_tot = new long long[size];
        arr_A[0] = local_A;
        arr_tot[0] = local_validas;

        long long total_A = local_A;
        long long total_validas = local_validas;

        for (int p = 1; p < size; p++) {
            MPI_Recv(&arr_A[p], 1, MPI_LONG_LONG, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&arr_tot[p], 1, MPI_LONG_LONG, p, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_A += arr_A[p];
            total_validas += arr_tot[p];
        }

        for (int p = 0; p < size; p++) {
            printf("Proceso %d -> A: %lld | Bases validas: %lld\n", p, arr_A[p], arr_tot[p]);
        }

        double porcentaje_A = ((double)total_A / (double)total_validas) * 100.0;
        printf("Total procesos: %d | Bases validas: %lld | Total A: %lld | Porcentaje A: %.4f%%\n",
               size, total_validas, total_A, porcentaje_A);

        mkdir("results", 0777);
        FILE *fp = fopen("results/ejercicio4_procesos.csv", "w");
        if (fp) {
            fprintf(fp, "proceso,bases_A,bases_validas\n");
            for (int p = 0; p < size; p++) {
                fprintf(fp, "%d,%lld,%lld\n", p, arr_A[p], arr_tot[p]);
            }
            fclose(fp);
        }

        delete[] arr_A;
        delete[] arr_tot;
    } else {
        MPI_Send(&local_A, 1, MPI_LONG_LONG, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&local_validas, 1, MPI_LONG_LONG, 0, 3, MPI_COMM_WORLD);
    }

    delete[] bloque;
    MPI_Finalize();
    return 0;
}
