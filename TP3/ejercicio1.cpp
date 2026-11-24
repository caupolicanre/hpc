#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>


double sumar_por_filas(const double *matriz, long long N, int num_threads)
{
    double suma = 0.0;
    omp_set_num_threads(num_threads);

    #pragma omp parallel for reduction(+:suma) default(none) shared(matriz, N)
    for (long long i = 0; i < N; i++) {
        long long offset = i * N;
        for (long long j = 0; j < N; j++) {
            suma += matriz[offset + j];
        }
    }
    return suma;
}

double sumar_por_columnas(const double *matriz, long long N, int num_threads)
{
    double suma = 0.0;
    omp_set_num_threads(num_threads);

    #pragma omp parallel for reduction(+:suma) default(none) shared(matriz, N)
    for (long long j = 0; j < N; j++) {
        for (long long i = 0; i < N; i++) {
            suma += matriz[i * N + j];
        }
    }
    return suma;
}

int main()
{
    const long long N = 12000;
    const long long total_elem = N * N;
    const int n_pruebas = 5;
    const int hilos[n_pruebas] = {1, 2, 4, 8, 16};
    const int repeticiones = 3;

    double *matriz = new double[total_elem];
    #pragma omp parallel for default(none) shared(matriz, total_elem)
    for (long long idx = 0; idx < total_elem; idx++) {
        matriz[idx] = 1.0;
    }

    // Warmup
    double s_warm = sumar_por_filas(matriz, N, 4);
    (void)s_warm;

    mkdir("results", 0777);
    FILE *fp = fopen("results/ejercicio1_suma_matriz.csv", "w");
    if (fp) {
        fprintf(fp, "threads,t_filas_s,t_columnas_s,speedup_filas,speedup_columnas,eficiencia_filas_pct,eficiencia_columnas_pct,ratio_col_vs_fila\n");
    }

    double t1_filas = 0.0;
    double t1_cols = 0.0;

    printf("Dimension de la matriz: %lld x %lld (%lld elementos)\n", N, N, total_elem);

    for (int k = 0; k < n_pruebas; k++) {
        int th = hilos[k];

        double acum_filas = 0.0;
        double res_filas = 0.0;
        for (int r = 0; r < repeticiones; r++) {
            double t0 = omp_get_wtime();
            res_filas = sumar_por_filas(matriz, N, th);
            double t1 = omp_get_wtime();
            acum_filas += (t1 - t0);
        }
        double t_filas = acum_filas / repeticiones;

        double acum_cols = 0.0;
        double res_cols = 0.0;
        for (int r = 0; r < repeticiones; r++) {
            double t0 = omp_get_wtime();
            res_cols = sumar_por_columnas(matriz, N, th);
            double t1 = omp_get_wtime();
            acum_cols += (t1 - t0);
        }
        double t_cols = acum_cols / repeticiones;

        if (th == 1) {
            t1_filas = t_filas;
            t1_cols = t_cols;
        }

        double sp_filas = t1_filas / t_filas;
        double sp_cols = t1_cols / t_cols;
        double ef_filas = (sp_filas / th) * 100.0;
        double ef_cols = (sp_cols / th) * 100.0;
        double ratio = t_cols / t_filas;

        printf("Threads: %2d | Filas: %.6f s (S=%.2fx, E=%.1f%%) | Columnas: %.6f s (S=%.2fx, E=%.1f%%) | Ratio Col/Fila: %.2fx | Suma: %.0f\n",
               th, t_filas, sp_filas, ef_filas, t_cols, sp_cols, ef_cols, ratio, (res_filas + res_cols) / 2.0);

        if (fp) {
            fprintf(fp, "%d,%.6f,%.6f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
                    th, t_filas, t_cols, sp_filas, sp_cols, ef_filas, ef_cols, ratio);
        }
    }

    if (fp) fclose(fp);
    delete[] matriz;
    return 0;
}
