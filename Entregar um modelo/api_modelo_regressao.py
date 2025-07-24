from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import uvicorn

# Carrega o modelo
modelo = joblib.load("./modelo_regressao_linear.pkl")

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
