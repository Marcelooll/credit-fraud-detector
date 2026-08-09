# FraudSentinel

FraudSentinel é um projeto completo de machine learning para detecção de anomalias em transações de cartão de crédito. Ele combina aquisição de dados, engenharia de features, treinamento de modelo não supervisionado e um painel interativo moderno construído com Streamlit.

Este projeto foi pensado para funcionar tanto como um excelente portfólio quanto como um exemplo prático de como construir um sistema de detecção de anomalias de forma explicável e aplicável.

A versão 1.1 inclui uma interface multilíngue no Streamlit, simulação de transações em tempo real, análise em lote via CSV, diagnósticos explicativos e compatibilidade pronta para deploy no Streamlit Cloud.

Se você encontrar bugs, tiver sugestões ou quiser solicitar alterações, por favor abra uma issue ou entre em contato com o mantenedor.

---

## 1. Deploy no Streamlit Cloud

Este projeto está preparado para deploy no Streamlit Cloud com a estrutura padrão:

- ponto de entrada: app.py
- dependências: requirements.txt
- runtime Python: runtime.txt
- configuração Streamlit: .streamlit/config.toml

Use esta URL após publicar o repositório:

https://share.streamlit.io/Marcelooll/credit-fraud-detector/main/app.py

## 2. Visão Geral do Projeto

FraudSentinel utiliza o algoritmo Isolation Forest para identificar transações suspeitas sem depender de rótulos de fraude durante o treinamento. O sistema é treinado com um dataset real de fraude em cartão de crédito e expõe os resultados por meio de uma aplicação web com:

- simulação de transação em tempo real;
- análise em lote via CSV;
- pontuação de risco;
- explicações legíveis para humanos;
- visualizações interativas.

O projeto representa um fluxo real de ML porque inclui:

1. aquisição de dados;
2. pré-processamento e engenharia de features;
3. treinamento do modelo;
4. inferência;
5. apresentação em uma interface simples e acessível.

---

## 2. Por que este projeto é interessante

Este projeto é relevante porque demonstra várias competências importantes:

- conceitos de aprendizado supervisionado e não supervisionado;
- engenharia de features a partir de dados transacionais;
- detecção de anomalias com Isolation Forest;
- implantação de um projeto de dados como aplicação web;
- comunicação clara de previsões em uma interface intuitiva.

Para um estudante ou profissional iniciante, isso é mais forte do que apenas mostrar um notebook, porque mostra que você consegue entregar um produto funcional.

---

## 3. Principais funcionalidades

- Simulação de risco de transação em tempo real
- Inferência em lote a partir de arquivos CSV
- Visualizações analíticas com Plotly
- Camada de explicação baseada em regras para cada previsão
- Histórico de sessão e exportação para CSV
- Interface moderna e responsiva com Streamlit
- Artefatos do modelo persistidos localmente para reutilização rápida

---

## 4. Ferramentas utilizadas

### Stack principal

- Python 3.10+
- Pandas para manipulação de dados
- NumPy para operações numéricas
- Scikit-learn para Isolation Forest e pré-processamento
- Joblib para serialização do modelo
- Streamlit para a interface web interativa
- Plotly para gráficos e dashboards

### Dados e suporte ao ML

- KaggleHub para download do dataset
- StandardScaler para normalização das features
- IsolationForest para detecção de anomalias

### Dependências adicionais

- SHAP
- imbalanced-learn

Essas bibliotecas foram incluídas para apoiar experimentação e evolução futura do projeto, embora o funcionamento principal dependa da stack acima.

---

## 5. Arquitetura do projeto

O sistema segue uma estrutura simples e modular:

```text
Dados brutos de transação
        │
        ▼
Engenharia de features
        │
        ▼
Pré-processamento e escala
        │
        ▼
Treinamento do Isolation Forest
        │
        ▼
Artefatos do modelo (.pkl)
        │
        ▼
Dashboard com Streamlit
        ├── Previsão em tempo real
        ├── Processamento em lote via CSV
        └── Visualizações e explicações
```

### Componentes

- train.py: prepara os dados, cria features, treina o modelo, avalia o desempenho e salva os artefatos.
- app.py: carrega o modelo treinado, recebe entradas do usuário ou arquivos CSV, executa inferência e renderiza o painel.
- model/: armazena o modelo treinado, o scaler e a lista de features.
- requirements.txt: contém as dependências Python necessárias para rodar o projeto.

---

## 6. Estrutura do projeto

```text
credit-fraud-detector/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── README.pt-BR.md
└── model/
    ├── isolation_forest.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

---

## 7. Pré-requisitos

Antes de rodar o projeto localmente, garanta que você tenha:

- Python 3.10 ou superior
- pip instalado
- uma conta no Kaggle
- credenciais da API do Kaggle configuradas

### Como configurar as credenciais do Kaggle

1. Acesse https://www.kaggle.com/settings
2. Abra a seção API
3. Crie um novo token da API
4. Salve o arquivo kaggle.json baixado em:
   - Windows: C:\Users\SEU_USUARIO\.kaggle\kaggle.json
   - macOS/Linux: ~/.kaggle/kaggle.json

Se as credenciais não estiverem corretas, o script de treinamento não conseguirá baixar o dataset.

---

## 8. Instalação e configuração

### Windows

```bash
cd credit-fraud-detector
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
cd credit-fraud-detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 9. Como rodar o projeto

### Passo 1: Treinar o modelo

```bash
python train.py
```

Esse script irá:

- baixar o dataset do Kaggle;
- criar features como hora, dia da semana, idade e valor transformado com log;
- escalar os dados;
- treinar o Isolation Forest;
- salvar os artefatos na pasta model/.

### Passo 2: Iniciar a aplicação

```bash
streamlit run app.py
```

Em seguida, abra a URL local mostrada no terminal, normalmente:

```text
http://localhost:8501
```

---

## 10. Como usar a aplicação

### Aba de simulação ao vivo

Use este modo para testar uma transação por vez.

Você pode ajustar valores como:

- valor da transação;
- hora da transação;
- dia da semana;
- idade do titular;
- população da cidade;
- coordenadas do estabelecimento;
- coordenadas do titular.

A aplicação retornará:

- uma pontuação de risco;
- um veredito como seguro ou suspeito;
- uma explicação em linguagem simples;
- gráficos diagnósticos interativos.

### Aba de processamento em lote

Use esta aba para fazer upload de um arquivo CSV e executar inferência em várias linhas ao mesmo tempo.

Você pode:

- enviar um dataset customizado;
- testar com uma amostra sintética;
- analisar os resultados em tabela;
- baixar as previsões em CSV.

### Aba de insights

Esta seção ajuda a entender melhor o comportamento do modelo, incluindo:

- parâmetros do modelo;
- teoria da detecção de anomalias;
- visualizações da distribuição dos resultados.

---

## 11. Lógica do modelo

O projeto usa uma abordagem de detecção de anomalias não supervisionada.

Por que Isolation Forest?

- não exige exemplos rotulados de fraude para detectar anomalias;
- é adequado para eventos raros, como fraude;
- gera scores de anomalia que podem ser interpretados como sinal de risco.

O modelo produz uma pontuação que indica o quanto uma transação é incomum em relação à linha de base aprendida. Na aplicação, essa pontuação é traduzida para uma visão de risco mais compreensível para o usuário final.

---

## 12. Deploy no Streamlit Community Cloud

Sim — este projeto é muito adequado para deploy no Streamlit Community Cloud.

### Fluxo recomendado de deploy

1. Envie o repositório para o GitHub.
2. Acesse o Streamlit Community Cloud.
3. Crie um novo app.
4. Selecione o repositório e a branch.
5. Defina o arquivo principal como app.py.
6. Faça o deploy.

### Pontos importantes

- A aplicação usa os arquivos do modelo localizados na pasta model/, então eles precisam estar presentes no repositório antes do deploy.
- Mantenha segredos, como credenciais do Kaggle, fora do repositório.
- Para um deployment mais robusto, você pode evoluir depois para um fluxo de gestão de ambiente mais profissional.

Isso torna o projeto especialmente atrativo para portfólio porque demonstra não só desenvolvimento de modelo, mas também maturidade para publicação.

---

## 13. Pontos fortes para entrevistas

Se você quiser usar este projeto em entrevistas ou no currículo, esses são pontos fortes para destacar:

- construiu um pipeline completo de machine learning do dado até a interface;
- implementou uma solução de detecção de anomalias não supervisionada;
- trabalhou com dados reais de transações e engenharia de features;
- publicou uma aplicação de dados com Streamlit;
- criou uma experiência de usuário com explicabilidade e visualização.

---

## 14. Próximos passos

Possíveis melhorias futuras para o projeto:

- adicionar explicabilidade com SHAP;
- melhorar a calibração do modelo;
- incluir heurísticas mais avançadas de fraude;
- integrar um banco de dados para histórico de transações;
- criar uma API para inferência do modelo;
- adicionar testes automatizados.

---

## 15. Licença

Este projeto está disponível para uso educacional e pessoal.

Se quiser, também é possível adaptar o projeto para um cenário mais formal acadêmico ou comercial.
