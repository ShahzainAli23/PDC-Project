#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/sha.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>

#define PASSWORD_LENGTH 5
#define ALPHANUM "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_SIZE 62

char *charset = ALPHANUM;
char target_password[PASSWORD_LENGTH + 1];
unsigned char target_hash[SHA256_DIGEST_LENGTH];

void sha256(const char *str, unsigned char output[SHA256_DIGEST_LENGTH]) {
    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    SHA256_Update(&ctx, str, strlen(str));
    SHA256_Final(output, &ctx);
}

int compare_hash(const unsigned char *hash1, const unsigned char *hash2) {
    return memcmp(hash1, hash2, SHA256_DIGEST_LENGTH) == 0;
}

void generate_random_password() {
    srand(time(NULL));
    for (int i = 0; i < PASSWORD_LENGTH; ++i)
        target_password[i] = charset[rand() % CHARSET_SIZE];
    target_password[PASSWORD_LENGTH] = '\0';
    sha256(target_password, target_hash);
}

int brute_force(unsigned char *hash, char *found, long start, long end, volatile int *stop_flag) {
    char guess[PASSWORD_LENGTH + 1];
    for (long i = start; i < end; ++i) {
        if (*stop_flag) break;

        long n = i;
        for (int j = PASSWORD_LENGTH - 1; j >= 0; --j) {
            guess[j] = charset[n % CHARSET_SIZE];
            n /= CHARSET_SIZE;
        }
        guess[PASSWORD_LENGTH] = '\0';

        unsigned char test_hash[SHA256_DIGEST_LENGTH];
        sha256(guess, test_hash);

        if (compare_hash(hash, test_hash)) {
            strcpy(found, guess);
            *stop_flag = 1;
            return 1;
        }
    }
    return 0;
}

void run_sequential() {
    char found[PASSWORD_LENGTH + 1];
    long max = 1;
    for (int i = 0; i < PASSWORD_LENGTH; ++i) max *= CHARSET_SIZE;

    double start = omp_get_wtime();
    volatile int dummy_flag = 0;
    brute_force(target_hash, found, 0, max, &dummy_flag);
    double end = omp_get_wtime();

    printf("Sequential Time: %.4f seconds\n", end - start);
}

void run_parallel() {
    char found[PASSWORD_LENGTH + 1] = {0};
    long max = 1;
    for (int i = 0; i < PASSWORD_LENGTH; ++i) max *= CHARSET_SIZE;
    volatile int password_found = 0;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        char local_found[PASSWORD_LENGTH + 1];
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        long chunk = max / num_threads;
        long s = thread_id * chunk;
        long e = (thread_id == num_threads - 1) ? max : s + chunk;

        if (brute_force(target_hash, local_found, s, e, &password_found)) {
            #pragma omp critical
            {
                strcpy(found, local_found);
                password_found = 1;
            }
        }
    }

    double end = omp_get_wtime();
    printf("Parallel (OpenMP) Time: %.4f seconds\n", end - start);
}

void run_distributed(int rank, int size) {
    char found[PASSWORD_LENGTH + 1] = {0};
    long max = 1;
    for (int i = 0; i < PASSWORD_LENGTH; ++i) max *= CHARSET_SIZE;

    long chunk = max / size;
    long start = rank * chunk;
    long end = (rank == size - 1) ? max : start + chunk;

    char local_found[PASSWORD_LENGTH + 1];
    int found_flag = 0;
    volatile int global_stop = 0;

    found_flag = brute_force(target_hash, local_found, start, end, &global_stop);

    if (found_flag) {
        MPI_Send(local_found, PASSWORD_LENGTH + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        MPI_Status status;
        MPI_Recv(found, PASSWORD_LENGTH + 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    }
}

int main(int argc, char **argv) {
    int rank, size;
    double dist_start, dist_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        generate_random_password();
        printf("Generated Password: %s\n", target_password);
        fflush(stdout);

        run_sequential();
        run_parallel();

        dist_start = omp_get_wtime();
    }

    MPI_Bcast(target_hash, SHA256_DIGEST_LENGTH, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
    run_distributed(rank, size);

    if (rank == 0) {
        dist_end = omp_get_wtime();
        printf("Distributed (MPI) Time: %.4f seconds\n", dist_end - dist_start);
    }

    MPI_Finalize();
    return 0;
}

