# Bibliotecas para lidar com os DataSets
import pandas as pd

# para carregar o modelo
import joblib


# carregando os dados para realizar previsao
df_previsao = pd.read_csv("df_previsao.csv")

# carregando o modelo para realizar a previsao
rf_regressor = joblib.load("random_forest_regressor.pkl")
previsao_gerada_modelo = rf_regressor.predict(df_previsao)

# salvar
previsao_gerada_modelo.to_csv("previsao_gerada_modelo.csv", index=False)
