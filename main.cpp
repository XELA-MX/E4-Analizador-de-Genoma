#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <genoma_modificado.txt> <genoma_original.txt>" << std::endl;
        return 1;
    }

    // Leer archivo de genoma modificado
    std::ifstream archivo(argv[1]);
    if (!archivo.is_open()) {
        std::cerr << "Error al abrir: " << argv[1] << std::endl;
        return 1;
    }

    // Leer todas las líneas del archivo y concatenarlas
    std::vector<std::string> lineas;
    std::string linea;
    while (std::getline(archivo, linea)) {
        lineas.push_back(linea);
    }
    archivo.close();

    // Concatenar todas las líneas en una sola cadena
    std::string concatenado;
    for (const auto& l : lineas) {
        concatenado += l;
    }

    // Leer archivo de referencia
    std::ifstream archivo_referencia(argv[2]);
    if (!archivo_referencia.is_open()) {
        std::cerr << "Error al abrir: " << argv[2] << std::endl;
        return 1;
    }

    // Leer todas las líneas del archivo de referencia y concatenarlas
    std::vector<std::string> lineas_referencia;
    std::string linea_referencia;
    while (std::getline(archivo_referencia, linea_referencia)) {
        lineas_referencia.push_back(linea_referencia);
    }
    archivo_referencia.close();

    // Concatenar todas las líneas de referencia en una sola cadena
    std::string concatenado_referencia;
    for (const auto& l : lineas_referencia) {
        concatenado_referencia += l;
    }


// Tamaño de las cadenas concatenadas
    const int n = concatenado.size();
    int contador = 0;

    // Buscar e imprimir mutaciones directamente sobre las cadenas
    for (int i = 0; i < n; ++i) {
        if (concatenado[i] != concatenado_referencia[i]) {
            std::cout << "Mutación en la posición " << i << std::endl;
            contador++;
        }
    }

    std::cout << "\nTotal mutaciones: " << contador << std::endl;

    return 0;
}
