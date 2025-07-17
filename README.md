# 🤖 Projeto da disciplina [IMD3001] - Introdução à Inteligência Artificial: Análise e Classificação do Fluxo de Trânsito

Este projeto realiza a análise inteligente do fluxo de veículos em tempo real, utilizando visão computacional e modelos de Inteligência Artificial para detectar, rastrear, contar e classificar a intensidade do trânsito.

## 📝 Visão geral do projeto

A análise manual do fluxo de veículos é um processo ineficiente e pouco escalável. Detectores simples podem contar veículos, mas falham em compreender o comportamento complexo do trânsito. Para resolver esse problema, este projeto implementa um sistema automatizado capaz de:

* **Identificar e rastrear** veículos de forma única e persistente.
* **Realizar uma contagem precisa**, evitando a dupla contagem de um mesmo veículo.
* **Classificar a intensidade do trânsito** em três níveis: Leve, Moderado ou Alto.

Para isso, o sistema utiliza dois modelos de classificação (Bayesiano e Cadeias de Markov) e compara suas abordagens na interpretação dos mesmos dados de tráfego.

## 🏗️ Arquitetura do sistema

O projeto é modularizado em três scripts principais, cada um com uma responsabilidade:

1.  **`contador_veiculos.py`**: É o coração da visão computacional. Ele utiliza o modelo **MobileNet SSD** com **OpenCV** para detectar e rastrear os veículos em cada quadro do vídeo. Este módulo gera os dados brutos de contagem de carros e motos que alimentam os outros componentes.
2.  **`classificador_bayesiano.py`**: Implementa um classificador que analisa o trânsito de forma instantânea, com base em evidências como a contagem de veículos e a hora do dia. Sua principal característica é ser altamente reativo a mudanças.
3.  **`classificador_markov.py`**: Oferece uma abordagem de classificação que leva em conta não apenas os dados atuais, mas também o estado anterior do trânsito , resultando em uma análise mais suave e estável.

## 🔍 Como funciona

### Detecção e rastreamento de veículos

* **Modelo de Detecção**: Utilizamos o **MobileNet SSD**, um modelo de *Deep Learning* pré-treinado, escolhido por ser rápido e eficiente para detecção de objetos em tempo real.
* **Lógica de Rastreamento**: Para "seguir" um veículo, o sistema associa um ID único a cada objeto detectado.
* **Veículo Conhecido**: O sistema utiliza um método de busca para encontrar a detecção mais próxima (dentro de uma distância `MAX_DISTANCE`) para manter o mesmo ID, efetivamente "seguindo" o veículo.

### Contagem precisa e sem duplicidade

Para garantir que cada veículo seja contado apenas uma vez, duas condições precisam ser atendidas simultaneamente:

1. O veículo precisa ser detectado de forma estável por um número mínimo de frames, definido pela variável `STABILITY_THRESHOLD`.
2. O veículo não pode ter sido contado antes (a flag `data['counted']` deve ser `False`).

### 🔧 Parâmetros configuráveis do rastreador

O comportamento do rastreador pode ser ajustado por meio de quatro variáveis principaiS:

| Variável | Descrição |
| :--- | :--- |
| `CONFIDENCE_THRESHOLD` | Confiança mínima para que uma detecção seja considerada válida. Ajuda a filtrar ruídos e falsos positivos. |
| `MAX_DISTANCE` | A distância máxima (em pixels) para associar uma nova detecção a um ID de veículo já existente. Define o "raio de busca". |
| `STABILITY_THRESHOLD` | Número de frames em que um veículo precisa ser visto de forma consistente antes de ser oficialmente contado. |
| `DISAPPEARED_THRESHOLD` | Número de frames que o sistema "espera" por um veículo que desapareceu temporariamente antes de removê-lo da memória. |

## 👷 Alunos

* [Gabriel Costa Lima Dantas](https://github.com/Gcld) 
* [Hugo Vinicius da Silva Figueirêdo](https://github.com/HugoViniciusSF)

## ⚙️ Requisitos

* 🐍 Ter o **Python 3** instalado na sua máquina.
* Instalar as bibliotecas necessárias:

```bash
pip install opencv-python
pip install opencv-python-headless
pip install opencv-contrib-python
pip install imutils
pip install numpy
pip install pandas
```
# 📊 Análise

Esta seção aprofunda os aspectos teóricos do agente inteligente, com base no material da apresentação.

## Algoritmos

O sistema emprega múltiplos algoritmos e formas de representação de conhecimento:

### Algoritmos utilizados:

* **Redes Neurais Profundas (Deep Learning):** O modelo `MobileNet SSD` é usado para a tarefa de detecção de objetos.
* **Inferência Bayesiana:** Um classificador Bayesiano é usado para determinar a intensidade do trânsito de forma instantânea, com base em evidências como a contagem de veículos e a hora do dia.
* **Cadeias de Markov:** Um modelo de Markov classifica o trânsito considerando o estado anterior, o que gera uma análise mais estável e suave.

### Métodos de busca:

O rastreamento de veículos implementa uma busca por vizinho mais próximo. Para um veículo já conhecido, o sistema procura a detecção mais próxima dentro de um raio de busca (`MAX_DISTANCE`) para manter sua identidade.

### Classificadores:

* No classificador Bayesiano, o conhecimento é representado por uma **Tabela de Probabilidade Condicional (CPT)**, que mapeia as evidências (contagem e hora) para a probabilidade de cada estado do trânsito.
* No classificador de Markov, o conhecimento é modelado como uma **máquina de estados**, onde as transições entre os estados (Leve, Moderado, Alto) dependem da observação atual (contagem de veículos) e do estado anterior.


## Fluxo do sistema:



## Modelagem do agente (PEAS e arquitetura)

* **Performance (Medida de Desempenho):** Contagem precisa de veículos, evitando duplicidade, e a correta classificação da intensidade do trânsito.
* **Environment (Ambiente):** Fluxo de tráfego em uma via, capturado por vídeo.
* **Actuators (Atuadores):** As "ações" do agente são a atualização de contadores internos (carros, motos, total), a marcação de um veículo como "contado", e a emissão de uma classificação de trânsito (Leve, Moderado, Alto).
* **Sensors (Sensores):** Câmera ou arquivo de vídeo, cujos frames são processados pelo OpenCV para extrair informações de pixels.

A **arquitetura do agente** é a de um **agente baseado em modelos**, pois ele mantém um estado interno sobre o mundo (a lista de veículos rastreados, seus IDs, posições e status de contagem) e utiliza modelos (Bayesiano e Markov) para tomar decisões sobre a classificação do trânsito.

# 📚 Referências

* OSEBROCK, Adrian. Object detection with deep learning and OpenCV. PylmageSearch, 11 set. 2017.
* RUSSELL, Stuart; NORVIG, Peter. Inteligência Artificial: Uma Abordagem Moderna. 4. ed. Rio de Janeiro: GEN | LTC, 2021.
* MARKOV, A.A. "Extension of the limit theorems of probability theory to a sum of variables connected in a chain". Reimpresso em: R. Howard. Dynamic Probabilistic Systems, volume 1: Markov Chains. John Wiley and Sons, 1971.
