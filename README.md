# F1 Champions Prediction

## English

Machine learning project for predicting Formula 1 race results and season standings. The application exposes a FastAPI API, uses trained XGBoost models, and transforms historical/current F1 data into model-ready features.

### Main features

- Race result prediction with either the `ranker` or `regressor` model.
- Race prediction by date using F1DB data.
- Driver and constructor season standings prediction.
- Supporting pipeline to load raw F1DB data, select historical context, build features, and map them into the API schema.
- Unit tests for loaders, selectors, feature engineering, mappers, services, and API helpers.

### Project structure

```text
config/                         Configuration files and mappings
models/                         Saved XGBoost models
notebooks/                      Data exploration, correction, and preparation
src/api/                        FastAPI app, routes, and DTOs
src/application/services/       Prediction and F1DB orchestration services
src/data/                       Data loaders and builders
src/features/                   Feature engineering
src/ml/                         XGBoost predictors
src/scripts/                    Helper scripts
src/test/                       Unit tests
train_and_test_data/            Training, test, and processed data
validation_data/                F1DB data used for validation/prediction
```

### Requirements

- Python `>=3.14`
- `uv` for environment setup and command execution

The main dependencies are listed in `pyproject.toml`: FastAPI, Uvicorn, pandas, numpy, scikit-learn, XGBoost, Pydantic, requests, matplotlib, JupyterLab, and coverage.

### Installation

```powershell
uv sync
```

### Run the API

In PowerShell:

```powershell
$env:PYTHONPATH = "src"
uv run uvicorn api.main:app --reload
```

Then open:

- Health check: `http://localhost:8000/health`
- Swagger docs: `http://localhost:8000/docs`

### Endpoints

- `POST /api/v1/predict/race?model_type=ranker`
  - Receives complete grid, driver, constructor, and circuit data.
  - Returns the predicted race classification.

- `POST /api/v1/predict/race-by-date?model_type=ranker`
  - Receives a race date as `YYYY-MM-DD`.
  - Uses F1DB data from `validation_data/01_raw`.
  - If qualifying data is available, it builds a post-qualifying payload; otherwise it uses the pre-qualifying model.

- `POST /api/v1/predict/season`
  - Receives the season year.
  - Can use current official results when `use_current_results` is `true`.
  - Returns predicted driver and constructor standings.

### Payload examples

Race prediction by date:

```json
{
  "race_date": "2026-07-26"
}
```

Season prediction:

```json
{
  "year": 2026,
  "use_current_results": true,
  "current_form_weight": 0.25
}
```

### Tests

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest discover -s src/test
```

The file `src/test/test_f1db_supporting_services.py` covers the F1DB supporting services, including:

- raw CSV loading and date conversion;
- historical data selection up to the requested race;
- target feature creation for a race;
- current-season features;
- mapping into API DTOs;
- `predict_service` behavior for races with and without qualifying data;
- manual API call helpers.

### Data and models

- Trained models are stored in `models/`.
- Historical training and test data are stored in `train_and_test_data/`.
- F1DB validation and date-based prediction data are stored in `validation_data/01_raw/`.
- `config/mapping_circuit_and_constructor.json` maps constructors and circuits to model IDs.

---

## Português

Projeto de machine learning para prever resultados de corridas e classificações de temporada da Fórmula 1. A aplicação expõe uma API FastAPI, usa modelos XGBoost já treinados e transforma dados históricos/atuais da F1 em features usadas pelos modelos.

### Principais recursos

- Predição de resultado de corrida com modelo `ranker` ou `regressor`.
- Predição de corrida por data usando dados F1DB.
- Predição de classificação de pilotos e construtores para uma temporada.
- Pipeline de suporte para carregar dados brutos F1DB, selecionar histórico, montar features e converter para o schema da API.
- Testes unitários para loaders, seletores, engenharia de features, mapeadores, serviços e helpers da API.

### Estrutura do projeto

```text
config/                         Arquivos de configuração e mapeamentos
models/                         Modelos XGBoost salvos
notebooks/                      Exploração, correção e preparação dos dados
src/api/                        Aplicação FastAPI, rotas e DTOs
src/application/services/       Serviços de predição e orquestração F1DB
src/data/                       Loaders e builders de dados
src/features/                   Engenharia de features
src/ml/                         Preditores XGBoost
src/scripts/                    Scripts auxiliares
src/test/                       Testes unitários
train_and_test_data/            Dados de treino, teste e processamento
validation_data/                Dados F1DB usados para validação/predição
```

### Requisitos

- Python `>=3.14`
- `uv` para instalação e execução do ambiente

As dependências principais estão em `pyproject.toml`: FastAPI, Uvicorn, pandas, numpy, scikit-learn, XGBoost, Pydantic, requests, matplotlib, JupyterLab e coverage.

### Instalação

```powershell
uv sync
```

### Executar a API

No PowerShell:

```powershell
$env:PYTHONPATH = "src"
uv run uvicorn api.main:app --reload
```

Depois acesse:

- Health check: `http://localhost:8000/health`
- Documentação Swagger: `http://localhost:8000/docs`

### Endpoints

- `POST /api/v1/predict/race?model_type=ranker`
  - Recebe os dados completos de grid, piloto, construtor e circuito.
  - Retorna a classificação prevista da corrida.

- `POST /api/v1/predict/race-by-date?model_type=ranker`
  - Recebe uma data de corrida no formato `YYYY-MM-DD`.
  - Usa os dados F1DB em `validation_data/01_raw`.
  - Se houver classificação disponível, cria o payload pós-qualifying; caso contrário, usa o modelo pre-qualifying.

- `POST /api/v1/predict/season`
  - Recebe o ano da temporada.
  - Pode usar resultados atuais quando `use_current_results` for `true`.
  - Retorna classificações previstas de pilotos e construtores.

### Exemplos de payload

Predição por data:

```json
{
  "race_date": "2026-07-26"
}
```

Predição de temporada:

```json
{
  "year": 2026,
  "use_current_results": true,
  "current_form_weight": 0.25
}
```

### Testes

```powershell
$env:PYTHONPATH = "src"
uv run python -m unittest discover -s src/test
```

O arquivo `src/test/test_f1db_supporting_services.py` cobre os serviços de apoio da integração F1DB, incluindo:

- leitura dos CSVs brutos e conversão de datas;
- seleção do histórico até a corrida solicitada;
- criação de features alvo para a corrida;
- features de temporada atual;
- mapeamento para DTOs da API;
- fluxo do `predict_service` para corridas com e sem classificação;
- helpers manuais de chamada da API.

### Dados e modelos

- Os modelos treinados ficam em `models/`.
- Os dados históricos de treino e teste ficam em `train_and_test_data/`.
- Os dados F1DB de validação e predição por data ficam em `validation_data/01_raw/`.
- O arquivo `config/mapping_circuit_and_constructor.json` mapeia construtores e circuitos para IDs usados pelo modelo.

---