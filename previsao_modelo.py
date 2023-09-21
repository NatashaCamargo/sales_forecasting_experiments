# Bibliotecas para lidar com os DataSets
import pandas as pd
import numpy as np

# para salvar o modelo
import joblib

#  Carregando o modelo
rf_regressor = joblib.load("random_forest_regressor.pkl")

# --PREVISAO DE QUANTIDADE DE VENDAS PARA CICLOS ESPECIFICADOS--
# Carregando os dados e ealizando a previsao para os
# ciclos 202016, 202017 e 202101
X_final = pd.read_csv("df_previsao.csv")
previsao_final = rf_regressor.predict(X_final)

# Fazendo a transformacao exponencial devido ao log transform
# utilizado no treino
previsao_final_transformada = np.exp(previsao_final)

# Conversao do NumPy array para DataFrame
previsao_df = pd.DataFrame(
    previsao_final_transformada, columns=["predictions"]
)

# Salvando a previsao final em um .xlsx
previsao_df.to_excel("previsao.xlsx", index=False)
