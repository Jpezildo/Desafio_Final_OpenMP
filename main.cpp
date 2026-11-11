#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <iomanip>

struct Classe {
    float limite_inferior;
    float limite_superior;
    float ponto_medio;
    int frequencia;
};

std::vector<Classe> agrupar_em_classes(const std::vector<float>& dados, float tamanho_intervalo) {
    if (dados.empty()) return {};

    float min_val = dados[0];
    float max_val = dados[0];
    for (float valor : dados) {
        if (valor < min_val) min_val = valor;
        if (valor > max_val) max_val = valor;
    }

    int num_classes = std::ceil((max_val - min_val) / tamanho_intervalo);
    if (num_classes == 0) num_classes = 1;

    std::vector<Classe> classes(num_classes);
    for (int i = 0; i < num_classes; ++i) {
        classes[i].limite_inferior = min_val + (i * tamanho_intervalo);
        classes[i].limite_superior = classes[i].limite_inferior + tamanho_intervalo;
        classes[i].ponto_medio = (classes[i].limite_inferior + classes[i].limite_superior) / 2.0f;
        classes[i].frequencia = 0;
    }

    for (float valor : dados) {
        int indice_classe = (valor - min_val) / tamanho_intervalo;
        if (indice_classe >= num_classes) {
            indice_classe = num_classes - 1;
        }
        classes[indice_classe].frequencia++;
    }

    return classes;
}


float calcular_media_seq(const std::vector<Classe>& classes, int total_pessoas) {
    float soma_produtos = 0.0f;
    int num_classes = classes.size();
    for (int i = 0; i < num_classes; ++i) {
        soma_produtos += classes[i].ponto_medio * classes[i].frequencia;
    }
    return soma_produtos / total_pessoas;
}

float calcular_desvio_padrao_seq(const std::vector<Classe>& classes, float media, int total_pessoas) {
    float soma_variancia = 0.0f;
    int num_classes = classes.size();
    for (int i = 0; i < num_classes; ++i) {
        float diferenca = classes[i].ponto_medio - media;
        soma_variancia += classes[i].frequencia * (diferenca * diferenca);
    }
    return std::sqrt(soma_variancia / total_pessoas);
}


float calcular_media_omp(const std::vector<Classe>& classes, int total_pessoas) {
    float soma_produtos = 0.0f;
    int num_classes = classes.size();
    #pragma omp parallel for reduction(+:soma_produtos)
    for (int i = 0; i < num_classes; ++i) {
        soma_produtos += classes[i].ponto_medio * classes[i].frequencia;
    }
    return soma_produtos / total_pessoas;
}

float calcular_desvio_padrao_omp(const std::vector<Classe>& classes, float media, int total_pessoas) {
    float soma_variancia = 0.0f;
    int num_classes = classes.size();
    #pragma omp parallel for reduction(+:soma_variancia)
    for (int i = 0; i < num_classes; ++i) {
        float diferenca = classes[i].ponto_medio - media;
        soma_variancia += classes[i].frequencia * (diferenca * diferenca);
    }
    return std::sqrt(soma_variancia / total_pessoas);
}


float calcular_coeficiente_variacao(float media, float desvio_padrao) {
    return (desvio_padrao / media) * 100.0f;
}

int main() {
    std::vector<float> estaturas;
    std::vector<float> pesos;
    float e, p;
    
    std::ifstream arquivo("dados.txt");
    if (!arquivo.is_open()) {
        std::cerr << "Erro: Nao foi possivel abrir o arquivo 'dados.txt'." << std::endl;
        return 1;
    }
    
    while (arquivo >> e >> p) {
        estaturas.push_back(e);
        pesos.push_back(p);
    }
    arquivo.close();

    int total_pessoas = estaturas.size();
    if (total_pessoas == 0) return 1;

 
    std::vector<Classe> classes_estatura = agrupar_em_classes(estaturas, 8.0f);
    std::vector<Classe> classes_peso = agrupar_em_classes(pesos, 4.0f);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Processando dados de " << total_pessoas << " pessoas...\n\n";

    
    double inicio_seq = omp_get_wtime();
    
    float media_estatura_seq = calcular_media_seq(classes_estatura, total_pessoas);
    float dp_estatura_seq = calcular_desvio_padrao_seq(classes_estatura, media_estatura_seq, total_pessoas);
    float cv_estatura_seq = calcular_coeficiente_variacao(media_estatura_seq, dp_estatura_seq);

    float media_peso_seq = calcular_media_seq(classes_peso, total_pessoas);
    float dp_peso_seq = calcular_desvio_padrao_seq(classes_peso, media_peso_seq, total_pessoas);
    float cv_peso_seq = calcular_coeficiente_variacao(media_peso_seq, dp_peso_seq);
    
    double fim_seq = omp_get_wtime();
    double tempo_seq = fim_seq - inicio_seq;

    
    double inicio_omp = omp_get_wtime();
    
    float media_estatura_omp = calcular_media_omp(classes_estatura, total_pessoas);
    float dp_estatura_omp = calcular_desvio_padrao_omp(classes_estatura, media_estatura_omp, total_pessoas);
    float cv_estatura_omp = calcular_coeficiente_variacao(media_estatura_omp, dp_estatura_omp);

    float media_peso_omp = calcular_media_omp(classes_peso, total_pessoas);
    float dp_peso_omp = calcular_desvio_padrao_omp(classes_peso, media_peso_omp, total_pessoas);
    float cv_peso_omp = calcular_coeficiente_variacao(media_peso_omp, dp_peso_omp);
    
    double fim_omp = omp_get_wtime();
    double tempo_omp = fim_omp - inicio_omp;

    
    std::cout << "=== RESULTADOS DA ESTATURA (Intervalo de 8 cm) ===\n";
    std::cout << "Media Ponderada: " << media_estatura_omp << " cm\n";
    std::cout << "Desvio Padrao:   " << dp_estatura_omp << " cm\n";
    std::cout << "Coef. Variacao:  " << cv_estatura_omp << "%\n\n";

    std::cout << "=== RESULTADOS DO PESO (Intervalo de 4 kg) ===\n";
    std::cout << "Media Ponderada: " << media_peso_omp << " kg\n";
    std::cout << "Desvio Padrao:   " << dp_peso_omp << " kg\n";
    std::cout << "Coef. Variacao:  " << cv_peso_omp << "%\n\n";

    std::cout << "=== ANALISE DE DESEMPENHO ===\n";
    std::cout << std::setprecision(6);
    std::cout << "Tempo Sequencial: " << tempo_seq << " segundos\n";
    std::cout << "Tempo Paralelo:   " << tempo_omp << " segundos\n";

    return 0;
}