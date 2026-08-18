# FraudSentinel — Motor Enterprise de Detecção de Anomalias

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg?style=for-the-badge&logo=streamlit)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg?style=for-the-badge&logo=plotly)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

O **FraudSentinel** é uma plataforma de detecção de anomalias em transações financeiras de alta performance baseada em Machine Learning não supervisionado. Desenvolvido sobre uma arquitetura de **Isolation Forest** treinada em mais de **1,29 milhão de transações reais**, o sistema identifica padrões emergentes de fraude e transações atípicas (*zero-day fraud*) em tempo real e em processamento em lote massivo, sem depender de rótulos históricos escassos, desbalanceados ou com latência de reporte (*chargebacks*).

---

## Arquitetura do Sistema e Fluxo de Dados

```text
                                  ┌────────────────────────────────┐
                                  │   Telemetria Bruta de Dados    │
                                  │   (Streaming ou Lote em CSV)   │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │ Pipeline Vetorizado de Features│
                                  │ - Delta de Distância Geodésica │
                                  │ - Transformação Logarítmica    │
                                  │ - Codificação Temporal 24h/7d  │
                                  │ - Estratificação Demográfica   │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │     StandardScaler Fit         │
                                  │  (Base Legítima de Referência) │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │  Isolation Forest (200 iTrees) │
                                  │  Cálculo de Comprimento Médio  │
                                  └───────────────┬────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────────┐
                                  │  Motor de Calibração de Decisão│
                                  │  - Pontuação Contínua de Risco │
                                  │  - Veredito Binário de Anomalia│
                                  │  - Diagnóstico Heurístico      │
                                  └───────────────┬────────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌────────────────────────────────┐                ┌────────────────────────────────┐
        │   Simulador em Tempo Real      │                │   Processador Massivo em Lote  │
        │   - Indicador de Risco e Radar │                │   - Análise Estatística        │
        │   - Diagnóstico Explicável     │                │   - Padrões Temporais/Demogr.  │
        │   - Persistência de Auditoria  │                │   - Exportação de CSV Completo │
        └────────────────────────────────┘                └────────────────────────────────┘
```

---

## Destaques de Engenharia e Machine Learning

### 1. Detecção Não Supervisionada de Anomalias
Classificadores supervisionados sofrem frequentemente com alto desbalanceamento de classes ($< 0,5\%$ de fraudes), deslocamento de distribuição (*concept drift*) e atraso no reporte de contestações bancárias (30 a 90 dias). O FraudSentinel contorna essas limitações modelando a densidade topológica das transações legítimas:
- **Particionamento Recursivo**: Subamostras do espaço de atributos são fatiadas por hiperplanos aleatórios. Transações anômalas, por apresentarem características discrepantes, exigem significativamente menos divisões (profundidade de árvore $h(x)$ menor) para serem isoladas.
- **Score de Anomalia Normalizado**:
  $$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$
  onde $c(n)$ é o comprimento médio de busca mal-sucedida em uma Árvore Binária de Busca (BST).
- **Taxa de Contaminação Calibrada**: Ajustada em $0,01$ ($1\%$) para otimizar o equilíbrio entre sensibilidade e taxa de falsos positivos.

### 2. Engenharia de Atributos Específica do Domínio
- **Cálculo de Delta Geodésico**: Mensura a distância espacial entre as coordenadas residenciais do titular $(\text{lat}, \text{long})$ e as coordenadas do terminal do estabelecimento comercial $(\text{merch\_lat}, \text{merch\_long})$.
- **Transformação Logarítmica de Montante**: Aplica $\log(1 + \text{amt})$ para estabilizar a variância e mitigar a assimetria característica de distribuições monetárias.
- **Ciclo Temporal Circadiano e Semanal**: Mapeia a hora do dia ($0\text{--}23\text{h}$) e o dia da semana para identificar anomalias em horários de pico noturno ou finais de semana.
- **Estratificação Demográfica**: Calcula a idade do titular dinamicamente a partir da data de nascimento em relação ao timestamp da transação e pondera pela densidade populacional da cidade.

---

## Resultados de Desempenho e Benchmarks

Treinado e avaliado sobre o corpus do **Kaggle Credit Card Fraud Detection** ($1.296.675$ registros):

| Métrica | Resultado | Contexto |
| :--- | :--- | :--- |
| **Pontuação ROC-AUC** | **0.90+** | Capacidade de ranqueamento da fronteira de decisão |
| **Latência de Inferência** | **$< 0,8\text{ ms}$ / transação** | Avaliação instantânea em chamadas unitárias |
| **Throughput em Lote** | **$> 12.000\text{ tx/s}$** | Inferência vetorizada em lote via NumPy e Scikit-Learn |
| **Escala da Base de Treino** | **1.296.675 registros** | Normalização ajustada estritamente sobre a linha de base legítima |

---

## Guia Prático de Testes (Como Testar Cada Funcionalidade)

Para validar a capacidade de inferência e a precisão dos diagnósticos na interface:

### Caso de Teste 1: Transação Nominal / Legítima (Safe Baseline)
* **Objetivo**: Verificar que compras de rotina são classificadas como seguras com baixo risco.
* **Valores de Entrada**:
  * **Valor da Transação**: `$45.50`
  * **Hora do Evento**: `14` (tarde) | **Dia da Semana**: `Terça-feira`
  * **Coordenadas do Titular**: `Lat: 40.7128`, `Long: -74.0060` (Nova York)
  * **Idade**: `35 anos` | **População da Cidade**: `500.000`
  * **Coordenadas do Estabelecimento**: `Lat: 40.7306`, `Long: -73.9352` (mesma região metropolitana)
* **Resultado Esperado**:
  * **Veredito**: `TRANSAÇÃO VERIFICADA // COMPORTAMENTO NOMINAL (SAFE)`
  * **Pontuação de Risco**: `< 25%` (Faixa Verde)
  * **Score Bruto**: `+0.15` a `+0.22`

### Caso de Teste 2: Anomalia Crítica / Fraude Extrema (High-Risk Anomaly)
* **Objetivo**: Forçar o colapso das árvores de isolamento com múltiplos fatores de risco simultâneos.
* **Valores de Entrada**:
  * **Valor da Transação**: `$9.850.00` (discrepância severa de montante)
  * **Hora do Evento**: `03` (madrugada crítica) | **Dia da Semana**: `Domingo`
  * **Coordenadas do Titular**: `Lat: 25.7617`, `Long: -80.1918` (Miami, FL)
  * **Idade**: `78 anos` | **População da Cidade**: `120` (área ultra-rural)
  * **Coordenadas do Estabelecimento**: `Lat: 64.2008`, `Long: -149.4937` (Fairbanks, Alasca — salto geográfico de mais de 6.000 km)
* **Resultado Esperado**:
  * **Veredito**: `ANOMALIA DETECTADA // RISCO ELEVADO DE FRAUDE (FRAUD)`
  * **Pontuação de Risco**: `> 85%` (Faixa Vermelha)
  * **Diagnóstico Heurístico**: Dispara alertas de pico de valor, horário crítico, anomalia geodésica e baixa densidade populacional.

### Caso de Teste 3: Processamento em Lote Massivo (Batch Engine)
* **Objetivo**: Avaliar a inferência vetorizada sobre múltiplos registros simultâneos.
* **Passo a Passo**:
  1. Acesse a aba `◈ [02] PROCESSADOR EM LOTE`.
  2. Clique no botão `Testar Amostra Sintética (30 registros)` ou faça o upload de um CSV próprio.
  3. Observe a atualização instantânea das métricas gerais, do gráfico de pizza e dos painéis de padrões temporais e dispersão Valor x Risco.
  4. Clique em `⤓ BAIXAR CSV COMPLETO COM PREDIÇÕES` para obter os dados classificados.

### Caso de Teste 4: Simulador Estocástico de Monte Carlo
* **Objetivo**: Testar a estabilidade da fronteira de decisão contra 50 transações geradas aleatoriamente.
* **Passo a Passo**:
  1. Acesse a aba `◈ [03] INSIGHTS & PARÂMETROS`.
  2. Clique em `Executar Teste Estocástico de Monte Carlo (50 amostras)`.
  3. Visualize o gráfico de dispersão gerado dinamicamente para inspecionar a separação entre *SAFE* e *FRAUD*.

### Caso de Teste 5: Persistência de Auditoria e Fuso Horário
* **Objetivo**: Garantir que as simulações são salvas com carimbo de tempo oficial e persistem após recarregar a página.
* **Passo a Passo**:
  1. Execute uma simulação na aba `◈ [01] SIMULAÇÃO AO VIVO`.
  2. Verifique a nova linha no topo do `◈ Log de Auditoria da Sessão` contendo o timestamp formatado (ex: `18/08/2026 18:30:00 (UTC-03:00)`).
  3. Recarregue a página (`F5`). Observe que todo o histórico e a telemetria lateral permanecem preservados via `data/simulation_history.json`.
  4. Teste a exportação em CSV ou a limpeza pelo botão `⟲ Limpar Auditoria`.

---

## Gargalos e Limitações Conhecidas

Como em qualquer sistema de aprendizado de máquina em produção, foram identificados pontos de atenção e gargalos técnicos:

1. **Hiperparâmetro de Contaminação Estático (`contamination=0.01`)**:
   * *Impacto*: O percentil pré-fixado assume uma proporção constante de 1% de anomalias no tráfego. Em datas sazonais com comportamento de consumo atípico (ex: Black Friday, Natal), esse limiar rígido pode gerar falsos positivos se não houver recalibração adaptativa.
2. **Inferência Pontual sem Memória de Estado (*Stateless Scoring*)**:
   * *Impacto*: O algoritmo avalia cada transação como um vetor isolado $\mathbf{x} \in \mathbb{R}^d$. Ele não rastreia velocidade transacional contínua (ex: 5 compras com o mesmo cartão em menos de 3 minutos em locais distintos).
3. **Aproximação Euclidiana de Distância Geodésica**:
   * *Impacto*: O cálculo de delta espacial entre latitude e longitude via produto escalar é ultrarrápido para vetorização, mas introduz distorção métrica em latitudes elevadas em comparação com o cálculo estrito da fórmula ortodrômica de *Haversine*.
4. **Concorrência de I/O em Arquivo Local de Auditoria**:
   * *Impacto*: O armazenamento do histórico em arquivo JSON local (`data/simulation_history.json`) é ideal para instâncias *single-tenant* ou demonstrações locais, mas apresenta risco de concorrência de escrita se a aplicação for escalada horizontalmente em múltiplos contêineres sem um banco de dados relacional ou chave-valor centralizado.

---

## Roadmap e Possíveis Melhorias

Para evolução em direção a uma arquitetura distribuída de nível bancário:

1. **Pipeline Híbrido em Ensemble (Isolation Forest + LightGBM/XGBoost)**:
   * Combinar a capacidade não supervisionada da Isolation Forest (detecção de novas fraudes sem rótulo) com árvores de decisão balanceadas treinadas supervisionadamente sobre contestações confirmadas.
2. **Streaming em Tempo Real com Feature Store (Apache Kafka + Apache Flink + Redis)**:
   * Implementação de janelas deslizantes dinâmicas para calcular métricas de velocidade: número de transações nas últimas 1h/24h, razão entre o valor atual e o gasto médio do cartão nos últimos 30 dias.
3. **Interpretabilidade Local Exata via SHAP (TreeSHAP)**:
   * Substituição das heurísticas de explicação por valores de Shapley exatos calculados diretamente sobre os nós da Isolation Forest, detalhando o impacto individual de cada atributo no score final.
4. **Desacoplamento em Microsserviços e Deploy Containerizado**:
   * Encapsulamento do motor de inferência em uma API assíncrona **FastAPI / gRPC** rodando em **Docker + Kubernetes**, monitorada com **Prometheus**, **Grafana** e **Evidently AI** para detecção contínua de desvio de dados (*Data Drift*).

---

## Estrutura do Repositório

```text
credit-fraud-detector/
├── .streamlit/
│   └── config.toml             # Configurações do servidor Streamlit
├── data/
│   └── simulation_history.json # Log de auditoria persistente das simulações
├── model/
│   ├── isolation_forest.pkl    # Modelo Isolation Forest serializado
│   ├── scaler.pkl              # Pipeline StandardScaler ajustado
│   └── feature_names.pkl       # Mapeamento do esquema de atributos
├── app.py                      # Aplicação principal do dashboard Streamlit
├── train.py                    # Pipeline de download, engenharia, treino e avaliação
├── requirements.txt            # Dependências do projeto
├── runtime.txt                 # Especificação da versão Python
├── README.md                   # Documentação técnica (Inglês)
└── README.pt-BR.md             # Documentação técnica (Português)
```

---

## Guia de Instalação e Execução Local

### Pré-requisitos
- Python 3.10 ou superior
- Git e Pip instalados

### 1. Instalação do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/Marcelooll/credit-fraud-detector.git
cd credit-fraud-detector

# Criar e ativar o ambiente virtual
# No Windows:
python -m venv .venv
.venv\Scripts\activate

# No macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 2. Treinamento do Modelo (Opcional — Pesos pré-treinados inclusos)

Para retreinar o modelo Isolation Forest a partir dos dados brutos do Kaggle:

```bash
# Garanta que suas credenciais da API do Kaggle (~/.kaggle/kaggle.json) estejam configuradas
python train.py
```

### 3. Inicialização da Aplicação

```bash
streamlit run app.py
```

Acesse a interface no navegador através do localhost.

---

## Licença

Este projeto está sob a licença **MIT**.
