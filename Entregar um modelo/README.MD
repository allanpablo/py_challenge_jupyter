## 🚀 API com FastAPI para Modelo de Regressão Linear
📁 Estrutura esperada
📦 projeto/
 ┣ 📄 modelo_regressao_linear.pkl
 ┣ 📄 main.py
📄 main.py
```
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Carrega o modelo
modelo = joblib.load("modelo_regressao_linear.pkl")

# Instância da API
app = FastAPI(title="API de Regressão Linear")

# Modelo de entrada (validação com Pydantic)
class Entrada(BaseModel):
    horas_estudo: float

# Rota principal
@app.post("/prever")
def prever_pontuacao(dados: Entrada):
    horas = [[dados.horas_estudo]]
    pontuacao = modelo.predict(horas)[0][0]
    return {"pontuacao_prevista": round(pontuacao, 2)}
```
▶️ Executar com Uvicorn
Execute no terminal:
uvicorn main:app --reload
Acesse: http://127.0.0.1:8000

Documentação automática: http://127.0.0.1:8000/docs
🔁 Exemplo de Requisição POST
curl -X 'POST' \
  'http://127.0.0.1:8000/prever' \
  -H 'Content-Type: application/json' \
  -d '{"horas_estudo": 30}'
Resposta:
{
  "pontuacao_prevista": 510.25
}
