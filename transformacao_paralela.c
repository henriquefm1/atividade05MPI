//Enrique Cipolla Martins - 10427834
//Henrique Ferreira Marciano - 10439797
//Pedro Gabriel - 10437465

//Compilar:  mpicc transformacao_paralela.c -o transformacao_paralela
//Executar:  mpirun -np 5 ./transformacao_paralela 
//(se faltar slots: mpirun --oversubscribe -np 5 ./transformacao_paralela)


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DATA_SIZE 100
#define NUM_PROCS 5

static void imprimir_vetor(const char *rotulo, int rank, const int *v, int n) {
    printf("[Processo %d] %s: [", rank, rotulo);
    for (int i = 0; i < n; i++) {
        printf("%d%s", v[i], (i < n - 1) ? ", " : "");
    }
    printf("]\n");
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // O programa exige exatamente 5 processos
    if (size != NUM_PROCS) {
        if (rank == 0) {
            fprintf(stderr, "Erro: execute com exatamente %d processos (mpirun -np %d ...)\n",
                    NUM_PROCS, NUM_PROCS);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int local_size = DATA_SIZE / size; // 20 elementos por processo

    int *dados_originais = NULL;   // só existe no rank 0
    int *dados_transformados = NULL;

    double inicio = MPI_Wtime();

#ifdef USE_ALLGATHER
    // Com Allgather, todos os processos precisam do buffer final
    dados_transformados = (int *)malloc(DATA_SIZE * sizeof(int));
#else
    if (rank == 0) {
        dados_transformados = (int *)malloc(DATA_SIZE * sizeof(int));
    }
#endif

    // Vetor original criado somente no rank 0
    if (rank == 0) {
        dados_originais = (int *)malloc(DATA_SIZE * sizeof(int));
        for (int i = 0; i < DATA_SIZE; i++) {
            dados_originais[i] = i + 1;
        }
        imprimir_vetor("Vetor original", rank, dados_originais, DATA_SIZE);
    }

    // Buffers locais
    int *parte_local = (int *)malloc(local_size * sizeof(int));
    int *parte_transformada = (int *)malloc(local_size * sizeof(int));

    // 1) Distribui partes iguais do vetor entre os processos
    MPI_Scatter(dados_originais, local_size, MPI_INT,
                parte_local, local_size, MPI_INT,
                0, MPI_COMM_WORLD);

    // 2) Transformação local: f(x) = x * x
    for (int i = 0; i < local_size; i++) {
        parte_transformada[i] = parte_local[i] * parte_local[i];
    }

    // 3) Reúne os resultados
#ifdef USE_ALLGATHER
    MPI_Allgather(parte_transformada, local_size, MPI_INT,
                  dados_transformados, local_size, MPI_INT,
                  MPI_COMM_WORLD);
#else
    MPI_Gather(parte_transformada, local_size, MPI_INT,
               dados_transformados, local_size, MPI_INT,
               0, MPI_COMM_WORLD);
#endif

    double fim = MPI_Wtime();

    // 4) Exibição do resultado
#ifdef USE_ALLGATHER
    // Todos têm o vetor final; imprime em ordem de rank para não embaralhar a saída
    for (int r = 0; r < size; r++) {
        if (rank == r) {
            imprimir_vetor("Vetor transformado", rank, dados_transformados, DATA_SIZE);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
#else
    if (rank == 0) {
        imprimir_vetor("Vetor transformado", rank, dados_transformados, DATA_SIZE);
    }
#endif

    if (rank == 0) {
        printf("[Processo 0] Tempo de execução: %f s\n", fim - inicio);
    }

    free(parte_local);
    free(parte_transformada);
    free(dados_originais);       // free(NULL) é seguro
    free(dados_transformados);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
