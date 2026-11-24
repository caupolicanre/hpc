#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>

/*
- Variables shared:
    limite_max: solo lectura por todos los hilos para conocer la cota superior de busqueda.
- Variables private:
    n (indice del bucle paralelo) y las variables internas de es_primo (d, limite):
    cada hilo evalua distintos candidatos de forma independiente y necesita su propio estado.
- Variables Reduccion (reduction(+:total_primos)):
    crea una copia privada inicializada en 0 por cada hilo para evitar condiciones de carrera
    y al finalizar la region paralela suma todas las copias parciales en la variable global.
*/


bool es_primo(long long n)
{
    if (n <= 1) return false;
    if (n == 2 || n == 3) return true;
    if (n % 2 == 0) return false;

    long long limite = (long long)sqrt((double)n);
    for (long long d = 3; d <= limite; d += 2) {
        if (n % d == 0) return false;
    }
    return true;
}

double evaluar_esquema(long long limite_max, int num_threads, char tipo_sched, long long *conteo_out)
{
    omp_set_num_threads(num_threads);
    long long total_primos = 0;
    double t0 = omp_get_wtime();

    if (tipo_sched == 's') {
        #pragma omp parallel for reduction(+:total_primos) schedule(static) default(none) shared(limite_max)
        for (long long n = 2; n <= limite_max; n++) {
            if (es_primo(n)) total_primos++;
        }
    } else if (tipo_sched == 'd') {
        #pragma omp parallel for reduction(+:total_primos) schedule(dynamic, 1000) default(none) shared(limite_max)
        for (long long n = 2; n <= limite_max; n++) {
            if (es_primo(n)) total_primos++;
        }
    } else if (tipo_sched == 'g') {
        #pragma omp parallel for reduction(+:total_primos) schedule(guided) default(none) shared(limite_max)
        for (long long n = 2; n <= limite_max; n++) {
            if (es_primo(n)) total_primos++;
        }
    }

    double t1 = omp_get_wtime();
    *conteo_out = total_primos;
    return (t1 - t0);
}

int main()
{
    const long long limite_max = 10000000; // 1e7
    const int n_hilos = 5;
    const int hilos[n_hilos] = {1, 2, 4, 8, 16};
    const char codigos[3] = {'s', 'd', 'g'};
    const char *nombres[3] = {"static", "dynamic", "guided"};

    // Tiempo secuencial base (1 hilo)
    long long primos_seq = 0;
    double t0 = omp_get_wtime();
    for (long long n = 2; n <= limite_max; n++) {
        if (es_primo(n)) primos_seq++;
    }
    double t_seq = omp_get_wtime() - t0;

    printf("Busqueda de primos hasta %lld | Primos encontrados: %lld | Tiempo secuencial (1 hilo): %.6f s\n",
           limite_max, primos_seq, t_seq);

    mkdir("results", 0777);
    FILE *fp = fopen("results/ejercicio2_primos_omp.csv", "w");
    if (fp) {
        fprintf(fp, "schedule,threads,tiempo_s,speedup,eficiencia_pct,primos\n");
    }

    for (int s = 0; s < 3; s++) {
        for (int k = 0; k < n_hilos; k++) {
            int th = hilos[k];
            long long primos_hallados = 0;
            double t_par = (th == 1) ? t_seq : evaluar_esquema(limite_max, th, codigos[s], &primos_hallados);
            if (th == 1) primos_hallados = primos_seq;

            double speedup = t_seq / t_par;
            double eficiencia = (speedup / th) * 100.0;

            printf("Schedule: %-7s | Threads: %2d | Tiempo: %.6f s | Speedup: %.4fx | Eficiencia: %.2f%% | Primos: %lld\n",
                   nombres[s], th, t_par, speedup, eficiencia, primos_hallados);

            if (fp) {
                fprintf(fp, "%s,%d,%.6f,%.4f,%.2f,%lld\n",
                        nombres[s], th, t_par, speedup, eficiencia, primos_hallados);
            }
        }
    }

    if (fp) fclose(fp);
    return 0;
}
