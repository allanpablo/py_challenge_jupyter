from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Inicialização da aplicação
app = FastAPI(
    title="API de Regressão Linear",
    description="Predição de pontuação com base nas horas de estudo",
    version="1.0.0"
)

# Carregar modelo previamente treinado
modelo = joblib.load("modelo_regressao_linear.pkl")

# Definição do schema de entrada com Pydantic
class EntradaModelo(BaseModel):
    horas_estudo: float

# Definição da rota POST para predição
@app.post("/prever", summary="Retorna a pontuação prevista")
def prever(dados: EntradaModelo):
    entrada = [[dados.horas_estudo]]
    resultado = modelo.predict(entrada)[0][0]
    return {"pontuacao_prevista": round(resultado, 2)}
