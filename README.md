# Datathon - Bank Marketing

Projeto desenvolvido para o Datathon da Pós-graduação em Engenharia de Machine Learning da FIAP.

## 🎯 Objetivo

Desenvolver uma solução de experimentação adaptativa para recomendação de ofertas/ações de contato, utilizando algoritmos de Multi-Armed Bandit, com foco em Thompson Sampling.

O projeto utiliza o dataset Bank Marketing, disponibilizado originalmente pelo UCI Machine Learning Repository e utilizado no Kaggle.

Origem dos dados:
https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing

---

## ▶️ Video Apresentação
``` code

```

---

## 🌐 API em Produção
A API foi desenvolvida utilizando FastAPI e está disponível em produção no Render.
``` code
Rota inicial:  https://datathon-bank-marketing.onrender.com/
Documentação:  https://datathon-bank-marketing.onrender.com/docs
Status da API: https://datathon-bank-marketing.onrender.com/health
```

---

## 🚀 Funcionalidades

- Download e preparação do dataset Bank Marketing.
- Tratamento e transformação das variáveis.
- Criação de variável de recompensa (reward).
- Treinamento de modelos de classificação para os diferentes braços de decisão.
- Experimentação com Baseline.
- Experimentação com Thompson Sampling.
- Experimentação com Contextual Thompson Sampling.
- Experimentação com Bootstrap Thompson Sampling.
- Comparação dos métodos utilizando a taxa de conversão.
- Geração de um Golden Set para validação das recomendações.
- API REST para recomendar o melhor braço para cada cliente.
- Monitoramento básico da API através da rota /health.

Os braços considerados na experimentação são:
- cellular
- telephone

---

## 🧠 Estratégia de Recomendação

O problema é tratado como uma estratégia de Multi-Armed Bandit, na qual cada tipo de contato representa um braço de decisão.

Para cada cliente, o sistema utiliza as características disponíveis para estimar a probabilidade de conversão de cada braço e selecionar uma recomendação.

O Thompson Sampling utiliza amostragem das distribuições de recompensa para equilibrar:

- Exploração: testar diferentes opções para obter mais informações.
- Explotação: utilizar opções que apresentam melhores resultados observados.

No caso do Bootstrap Thompson Sampling, múltiplos modelos são treinados para cada braço, permitindo representar a incerteza das previsões durante o processo de decisão.

--- 
## 📊 Resultados da Experimentação

Os experimentos foram avaliados utilizando a **taxa de conversão** como principal métrica, considerando o conjunto de teste.

| Estratégia | Taxa de conversão |
|---|---:|
| Baseline — melhor braço fixo | **14,70%** |
| Thompson Sampling | **14,70%** |
| Contextual Thompson Sampling | **15,24%** |
| Bootstrap Thompson Sampling | **15,07%** |

Embora o Contextual Thompson Sampling tenha apresentado uma taxa de conversão superior no experimento, opto pelo Bootstrap Thompson Sampling por sua abordagem de representação da incerteza através de múltiplos modelos, tornando-o uma alternativa mais flexível para o cenário de experimentação adaptativa.

---
## 🧱 Arquitetura do Projeto

```
datathon-bank-marketing/
│
├── app/                  # API FastAPI
│   ├── api.py
│   ├── schemas.py
│   └── services.py
│
├── data/
│   ├── raw/              # Dados brutos
│   └── processed/        # Dados processados
│
├── models/               # Modelos treinados
│   │ Bootstrap
|   └── cellular/
│   └── telephone/
│
├── notebooks/
|   ├── 01_eda_bank_marketing_organizado.ipynb
│   └── 02_baseline_e_thompson.ipynb
│
├── src/
│   ├── datathon_bank_marketing/
|   |   ├── bandit/
|   │   |   └── thompson_sampling.py
|   |   ├── data/
│   │   |   ├── download.py
│   │   |   └── preprocess.py   
|   |   ├── pipeline/
|   │   |   └── train.py
│
├── tests/
|   ├── test_data_download.py
│   └── test_pipeline.py
|
├── .gitignore
├── README.md
├── poetry.lock
└── pyproject.toml
```

---

### ▶️ Executando a API manualmente

Caso queira **Rodar a API em sua máquina local**

1. Clonar o repositório:
   ```bash
   git clone https://github.com/Gabriel-limadev/datathon-bank-marketing.git
   cd datathon-bank-marketing
   
2. Instalar dependências:
   ```bash
   poetry install

3. Rodar API
   ```bash
   poetry run uvicorn app.api:app --reload

A API ficará disponivel localmente em: http://127.0.0.1:8000/docs

---

## 📄 Exemplos de Requests e Responses

### 🔹 Download dados
**Request**
```http
POST /download
```
**Response**
```code
{
  "message": "Dataset baixado e preparado com sucesso.",
  "rows": 41188,
  "columns": 21
}
```

### 🔹 Treino do modelo
**Request**
```http
POST /train
```
**Response**
```code
{
  "status": "success",
  "message": "Modelos retreinados e salvos com sucesso."
}

```
### 🔹 Recomendação de contato
**Request**
```http
POST /recommend
```
**Response**
```code
{
  "recommended_arm": "cellular",
  "estimated_probability": 0.10977823244281086
}
```

---

## 🧪 Testes

Para executar os testes automatizados:

``` code
poetry run pytest
```
Para executar o Ruff:
``` code
poetry run ruff check .
```
---

## ☁️ Arquitetura em Nuvem — AWS

Para uma futura arquitetura totalmente gerenciada em nuvem, o projeto pode ser escalado utilizando serviços da **AWS**.

- **Amazon S3:** armazenamento dos dados brutos, dados processados e modelos treinados.
- **AWS Glue / Amazon SageMaker:** processamento dos dados, treinamento e avaliação dos modelos.
- **Amazon ECR:** armazenamento das imagens Docker da aplicação.
- **Amazon ECS com Fargate:** execução da API FastAPI em containers, sem necessidade de gerenciar servidores.
- **FastAPI:** disponibilização da API para consumo das recomendações.

Arquitetura:

```text
Dataset
   │
   ▼
Amazon S3
   │
   ▼
AWS Glue / SageMaker
   │
   ├── Treinamento
   ├── Avaliação
   └── Modelos
          │
          ▼
      Amazon S3
          │
          ▼
      Amazon ECR
          │
          ▼
  Amazon ECS + Fargate
          │
          ▼
        FastAPI
          │
          ▼
       Usuário/API
```
---

## 📈 Escalabilidade Futura

- Docker
- CI/CD com GitHub Actions
- Banco de dados para histórico de previsões
- Implementação de novos algoritmos de Multi-Armed Bandit, como UCB e Epsilon-Greedy.
- Dashboards

---

## 👨‍💻 Autor

Gabriel Lima  
Pós-graduação em Engenharia de Machine Learning — FIAP



