#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>


inline double f(double x)
{
    return 4.0 / (1.0 + x * x);
}

double integrar_pi(long long num_pasos, int num_threads, char esquema, double *pi_out)
{
    double dx = 1.0 / (double)num_pasos;
    double suma = 0.0;
    omp_set_num_threads(num_threads);

    double t0 = omp_get_wtime();

    if (esquema == 'r') {
        // Esquema 1: reduction(+:suma)
        #pragma omp parallel for reduction(+:suma) default(none) shared(num_pasos, dx)
        for (long long i = 0; i < num_pasos; i++) {
            double x = (i + 0.5) * dx;
            suma += f(x);
        }
    } else if (esquema == 'p') {
        // Esquema 2: Acumulacion parcial por hilo + seccion critical al final
        #pragma omp parallel default(none) shared(num_pasos, dx, suma)
        {
            double suma_local = 0.0;
            #pragma omp for
            for (long long i = 0; i < num_pasos; i++) {
                double x = (i + 0.5) * dx;
                suma_local += f(x);
            }
            #pragma omp critical
            {
                suma += suma_local;
            }
        }
    } else if (esquema == 'c') {
        // Esquema 3: critical dentro del bucle por cada iteracion
        #pragma omp parallel for default(none) shared(num_pasos, dx, suma)
        for (long long i = 0; i < num_pasos; i++) {
            double x = (i + 0.5) * dx;
            double val = f(x);
            #pragma omp critical
            {
                suma += val;
            }
        }
    } else if (esquema == 'o') {
        // Esquema 4: ordered dentro del bucle por cada iteracion
        #pragma omp parallel for ordered default(none) shared(num_pasos, dx, suma)
        for (long long i = 0; i < num_pasos; i++) {
            double x = (i + 0.5) * dx;
            double val = f(x);
            #pragma omp ordered
            {
                suma += val;
            }
        }
    }

    double t1 = omp_get_wtime();
    *pi_out = suma * dx;
    return (t1 - t0);
}

int main()
{
    const long long num_pasos = 20000000;
    const int n_hilos = 5;
    const int hilos[n_hilos] = {1, 2, 4, 8, 16};
    const int n_esq = 4;
    const char codigos[n_esq] = {'r', 'p', 'c', 'o'};
    const char *nombres[n_esq] = {"reduction", "critical_parcial", "critical_iteracion", "ordered"};

    double pi_seq = 0.0;
    double dx = 1.0 / (double)num_pasos;
    double suma_seq = 0.0;
    double t0 = omp_get_wtime();
    for (long long i = 0; i < num_pasos; i++) {
        double x = (i + 0.5) * dx;
        suma_seq += f(x);
    }
    double t_seq = omp_get_wtime() - t0;
    pi_seq = suma_seq * dx;

    printf("Integracion numerica de Pi (%lld pasos) | Tiempo secuencial base: %.6f s | Pi = %.12f\n",
           num_pasos, t_seq, pi_seq);

    mkdir("results", 0777);
    FILE *fp = fopen("results/ejercicio4_pi_omp.csv", "w");
    if (fp) {
        fprintf(fp, "esquema,threads,tiempo_s,speedup,eficiencia_pct,pi_estimado\n");
    }

    for (int e = 0; e < n_esq; e++) {
        for (int k = 0; k < n_hilos; k++) {
            int th = hilos[k];
            double pi_est = 0.0;
            double t_par = integrar_pi(num_pasos, th, codigos[e], &pi_est);

            double speedup = t_seq / t_par;
            double eficiencia = (speedup / th) * 100.0;

            printf("Esquema: %-18s | Threads: %2d | Tiempo: %.6f s | Speedup: %.4fx | Eficiencia: %6.2f%% | Pi: %.10f\n",
                   nombres[e], th, t_par, speedup, eficiencia, pi_est);

            if (fp) {
                fprintf(fp, "%s,%d,%.6f,%.4f,%.2f,%.10f\n",
                        nombres[e], th, t_par, speedup, eficiencia, pi_est);
            }
        }
    }

    if (fp) fclose(fp);
    return 0;
}
