# Bibliotecas para lidar com os DataSets
import pandas as pd
import numpy as np

# importando ransom forest
from sklearn.ensemble import RandomForestRegressor

# Metricas para avaliacao
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import r2_score

# para salvar o modelo
import joblib

# --TREINO DO MODELO--
# carregando os csvs de treino
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")

# log transform do target
y_train = np.log(y_train)

# Instanciando o Random Forest
rf = RandomForestRegressor(n_estimators=1000, random_state=42)

rf.fit(X_train, y_train)

# --TESTANDO O MODELO--
# lendo os csvs de teste
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Realizando a previsao
y_pred_rf_ts = rf.predict(X_test)
y_test = np.log(y_test)

# MAE
mae_rf_ts = MAE(np.exp(y_test), np.exp(y_pred_rf_ts))
print("MAE RF serie temporal: {:.2f}".format(mae_rf_ts))

# R2
r2_rf_ts = r2_score(np.exp(y_test), np.exp(y_pred_rf_ts))
print("R-quadrado RF serie temporal:{:.2f}".format(r2_rf_ts))

# MAPE
mape_rs_ts = MAPE(np.exp(y_test), np.exp(y_pred_rf_ts))
print("MAPE  RF serie temporal: {:.2f}".format(mape_rs_ts))

# Salvando o modelo
joblib.dump(rf, "random_forest_regressor.pkl")
