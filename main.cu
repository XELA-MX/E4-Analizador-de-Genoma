#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Kernel para buscar mutaciones entre la secuencia y la referencia
__global__ void buscarMutaciones(char* secuencia, char* referencia, bool* mutaciones, int longitud) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < longitud) {
        mutaciones[idx] = (secuencia[idx] != referencia[idx]);
    }
}


// Kernel para imprimir las mutaciones encontradas y contar cuántas hay
__global__ void imprimirYContarMutaciones(bool* mutaciones, int longitud, int* contador) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < longitud && mutaciones[idx]) {
        printf("Mutación en la posición %d\n", idx);
        atomicAdd(contador, 1); // Incrementar el contador de mutaciones
    }
}

int main(int argc, char* argv[]) {
    // Verificiar que si se paso un archivo como argumento
    if (argc < 3) {
        std::cerr << "Uso: " << " <genoma_modificado.txt>" << " " << " <genoma_original.txt> " << std::endl;
        return 1;
    }

    // Intentar abrir el archivo
    std::ifstream archivo(argv[1]);
    if (!archivo.is_open()) {
        std::cerr << "Error al abrir: " << argv[1] << std::endl;
        return 1;
    }

    // Intentar abrir el archivo de referencia
    std::ifstream archivo_referencia(argv[2]);
    if (!archivo_referencia.is_open()) {
        std::cerr << "Error al abrir: " << argv[2] << std::endl;
        return 1;
    }

// Leer el archivo y almacenar las líneas en un vector
    std::vector<std::string> lineas;
    std::string linea;
    while (std::getline(archivo, linea)) {
        lineas.push_back(linea);
    }
    archivo.close();

    // Agregar líneas a concatenado
    std::string concatenado;
    for (const auto& l : lineas) {
        concatenado += l;
    }

    // Leer el archivo de referencia y almacenar las líneas en un vector
    std::vector<std::string> lineas_referencia;
    std::string linea_referencia;
    while (std::getline(archivo_referencia, linea_referencia)) {
        lineas_referencia.push_back(linea_referencia);
    }
    archivo_referencia.close();

    // Agregar líneas a concatenado
    std::string concatenado_referencia;
    for (const auto& l : lineas_referencia) {
        concatenado_referencia += l;
    }
    
    // Tmaño de n es el tamaño de la secuencia concatenada
    const int n = concatenado.size();

    // Reservar memoria en el host o CPU
    char* h_secuencia = concatenado.data();
    char* h_referencia = new char[n];  // Texto o genoma de referencia
    bool* h_mutaciones = new bool[n]; // Que mutaciones se han encontrado

    // copiar concatenado_referencia a h_referencia
    for (int i = 0; i < n; ++i) {
        h_referencia[i] = concatenado_referencia[i];
    }

    // Reservar memoria en el dispositivo o GPU
    char *d_secuencia, *d_referencia;
    bool *d_mutaciones;
    int *d_contador;
    int h_contador = 0;
    
    // Asignar memoria en el dispositivo o GPU
    cudaMalloc(&d_secuencia, n * sizeof(char));
    cudaMalloc(&d_referencia, n * sizeof(char));
    cudaMalloc(&d_mutaciones, n * sizeof(bool));
    cudaMalloc(&d_contador, sizeof(int));
    cudaMemcpy(d_secuencia, h_secuencia, n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_referencia, h_referencia, n * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_contador, &h_contador, sizeof(int), cudaMemcpyHostToDevice);

    // Valores eficientes para el grid y block
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    
    // Lanzar el kernel para buscar mutaciones
    buscarMutaciones<<<gridSize, blockSize>>>(d_secuencia, d_referencia, d_mutaciones, n);
    cudaDeviceSynchronize(); // Simplemente el await de CUDA

    imprimirYContarMutaciones<<<gridSize, blockSize>>>(d_mutaciones, n, d_contador); // Lanzar el kernel que solo cuenta mutaciones
    cudaDeviceSynchronize(); // Simplemente el await de CUDA

    cudaMemcpy(&h_contador, d_contador, sizeof(int), cudaMemcpyDeviceToHost); // Resultado del contador de mutaciones

    std::cout << "\nTotal mutaciones: " << h_contador << std::endl; // Imprimir el total de mutaciones encontradas

    // Liberar memoria
    delete[] h_referencia;
    delete[] h_mutaciones;
    
    cudaFree(d_secuencia);
    cudaFree(d_referencia);
    cudaFree(d_mutaciones);
    cudaFree(d_contador);

    return 0;
}
