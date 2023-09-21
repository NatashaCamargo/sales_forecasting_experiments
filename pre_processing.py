# Bibliotecas para lidar com os DataSets
import sqlite3
import pandas as pd
import numpy as np

# Box-cox transform
from scipy.stats import boxcox

# para realizar o train-test split
from sklearn.model_selection import train_test_split

"""
    Este codigo tem por objetivo realizar o tratamento dos dados
    e feature engineering.

    Ao final faremos o split para salvar em arquivos csv para treino, teste
    e previsao do modelo, os csvs ficaram armazenados para uso futuro caso
    haja interesse.
"""

# Conectando com a base e criando o dataframe
conn = sqlite3.connect("case_ds_gdem.sqlite3")
df = pd.read_sql_query("SELECT * FROM vendas", conn)

# criando uma copia do df para tratamento dos dados
df_to_split = df.copy()

# --MISSING VALUES--
"""
    Neste ponto sera feito o tratamento dos valores nulos. As seguintes
    variaveis serao tratadas:
        - VL_PRECO: os valores nulos serao substituidos pela media
        - PCT_DESCONTO: os valores nulos serao substituidos por zero
"""
# --VL_PRECO--
# Tratando o valor medio
val_medio = df["VL_PRECO"].mean()
# substituir valores pela media (treino e teste)
df_to_split["VL_PRECO"].fillna(val_medio, inplace=True)

# --PCT_DESCONTO--
# substituir os valores por zero
df_to_split["PCT_DESCONTO"].fillna(0, inplace=True)

#  --CAMPANHAS DE MARKETING--
"""
    As flags de campanha de marketing serao colocadas em apenas uma
    coluna que ira sinalizar a existencia das mesmas
"""
# listando as campanhas de marketing
campanhas_mkt = [
    "FLG_CAMPANHA_MKT_A",
    "FLG_CAMPANHA_MKT_B",
    "FLG_CAMPANHA_MKT_C",
    "FLG_CAMPANHA_MKT_D",
    "FLG_CAMPANHA_MKT_E",
    "FLG_CAMPANHA_MKT_F",
]


# criar uma nova coluna com valo 1 caso o item apresente alguma das
# campanhas de marketing, caso contrario 0
df_to_split["flg_mkt"] = np.where(
    (df_to_split[campanhas_mkt] != 0).any(axis=1), 1, 0
)
# eliminando as colunas de campanha
df_to_split.drop(columns=campanhas_mkt, inplace=True)

# --ENCODING DAS VARIAVEIS CATEGORICAS--
"""
    Aqui sera realizado o encoding das variaveis categoricas
    DES_CATEGORIA_MATERIAL e DES_MARCA_MATERIAL.
    Para isso utilizaremos a funcao encoding_categoria_produto
"""


def encoding_categoria_produto(df, var, target):
    """
    esta funcao tem o objetivo de realizar o encoding das variaveis
    categoricas o menor valor se refere a categoria com a menor media
    das quantidades vendidas
    Parametros:
        -df: pandas dataframe
        -var: variavel(eis) s ser(em) tratadas
        - target: target da analise
    """

    # ordenar as categorias da menor para a maior valor de
    # quantidades de itens vendidos
    labels_ordenadas = df.groupby([var])[target].mean().sort_values().index

    # criar um dicionario mapeando as categorias ordenadas
    ordinal_label = {k: i for i, k in enumerate(labels_ordenadas, 0)}

    # use the dictionary to replace the categorical strings by integers
    df[var] = df[var].map(ordinal_label)


# Realizando o encoding das variaveis categoricas
cat_vars = ["DES_CATEGORIA_MATERIAL", "DES_MARCA_MATERIAL"]
for var in cat_vars:
    encoding_categoria_produto(df_to_split, var, "QT_VENDA")

# --TRATAMENTO DA VARIAVEL TEMPORAL
"""
    Para a feature temporal COD_CICLO vamos seguir os seguintes passos:
        - Dividir em duas colunas, ano e ciclo
        - tanto ano quanto ciclo serão mapeados com números inteiros
    Isso sera feito com a funcao temporal_encoder
"""


def temporal_encoder(df, var_temporal):
    """
    Essa funcao cria duas novas colunas com o ano e o ciclo
    baseadas no conteudo de COD_CICLO.
    Parametros:
        - df: Pandas dataframe
        - var_temporal: variavel de tempo
    """
    # criando colunas ANO e CICLO
    df["ANO"] = df[var_temporal].astype(str).str[:4]
    df["CICLO"] = df[var_temporal].astype(str).str[-2:]

    # mapeando a variavel ano
    year_map = {"2018": 1, "2019": 2, "2020": 3, "2021": 3}
    df["ANO"] = df["ANO"].map(year_map)


# Tratando COD_CICLO
temporal_encoder(df_to_split, "COD_CICLO")

# --TRATAMENTO DA VARIAVEL NUMERICA VL_PRECO
"""
    A feature numérica VL_PRECO apresenta uma cauda longa para a direita,
    logo usaremos a transformação BOX-COX para obter uma distribuição mais
    próxima da normal.
"""
# Aplicando transformacao box-cox na variavel vl_preco
df_to_split["VL_PRECO_BOX_COX"], lam = boxcox(df_to_split["VL_PRECO"])

# --CRIANDO A COLUNA DA MEDIA MOVEL DE 3 PERIODOS
# criando uma copia do dataframe tratado e pronto para realizar os splits
df_media_movel_3 = df_to_split.copy()

# ordenando o dataframe port COD_MATERIAL e COD_CICLO em ordem crescente
df_media_movel_3.sort_values(
    by=["COD_MATERIAL", "COD_CICLO"], ascending=True, inplace=True
)


# Adicionar uma nova coluna QT_VENDA_lagged_3 que apresenta a media movel dos ultimos 3 periodos
df_media_movel_3["QT_VENDA_lagged_3"] = (
    df_media_movel_3.groupby("COD_MATERIAL")["QT_VENDA"]
    .rolling(window=3)
    .mean()
    .reset_index(0, drop=True)
)

# Substituindo os valores faltantes de 'QT_VENDA_lagged_3' com valores de 'QT_VENDA'
na_values = df_media_movel_3["QT_VENDA_lagged_3"].isna()
df_media_movel_3.loc[na_values, "QT_VENDA_lagged_3"] = df_media_movel_3.loc[
    na_values, "QT_VENDA"
]

# Para o step anterior como existem valores nulos para `QT_VENDA` para garantir que todos
# terao um valor iremos substutir pela media
median_QT_VENDA = df["QT_VENDA"].median()
df_media_movel_3["QT_VENDA_lagged_3"].fillna(median_QT_VENDA, inplace=True)

# --TRATAMENTO DOS VALORES EXTREMOS--
"""
    Como definido aqui iremos retirar do df os upper 1% tanto de QT_VENDA
    como de VL_PRECO, assim como tambem a retirada do outlier de PCT_DESCONTO
"""
# definindo upper 1% de QT_VENDA  e LV_PRECO
qt_venda_upper_1 = df["QT_VENDA"].quantile(0.99)
vl_preco_upper_1 = df["VL_PRECO"].quantile(0.99)

# Retirando upper 1%
df_media_movel_3_out = df_media_movel_3[
    (df_media_movel_3["QT_VENDA"] < qt_venda_upper_1)
    | (df_media_movel_3["QT_VENDA"].isnull())
].copy()
df_media_movel_3_out = df_media_movel_3_out[
    (df_media_movel_3_out["VL_PRECO"] < vl_preco_upper_1)
    | (df_media_movel_3_out["VL_PRECO"].isnull())
].copy()

# definindo o outlier a ser retirado de PCT_DESCONTO
pct_desconto_out = 8000
df_media_movel_3_out = df_media_movel_3_out[
    (df_media_movel_3_out["PCT_DESCONTO"] < pct_desconto_out)
    | (df_media_movel_3_out["PCT_DESCONTO"].isnull())
].copy()

# --CRIACAO DOS DF DE TREINO TESTE E PREVISAO--
# selecionando os dados do datarame inicial para serem feitas as previsoes ao final
df_previsao = df_media_movel_3_out[
    df_media_movel_3_out["COD_CICLO"].isin([202016, 202017, 202101])
]
# eliminando colunas que nao serao utilizadas na previsao
df_previsao = df_previsao.drop(
    ["COD_MATERIAL", "COD_CICLO", "VL_PRECO", "QT_VENDA"], axis=1
)

# Separando dataframes de treino e teste
df_train_test = df_media_movel_3_out[
    ~df_media_movel_3_out["COD_CICLO"].isin([202016, 202017, 202101])
]
X_train, X_test, y_train, y_test = train_test_split(
    df_train_test.drop(
        ["COD_MATERIAL", "COD_CICLO", "VL_PRECO", "QT_VENDA"], axis=1
    ),
    df_train_test["QT_VENDA"],  # target
    test_size=0.2,  # proporcao do dataset a ser alocado para teste
    random_state=42,  # seed
)

# --SALVANDO OS DATAFRAMES EM CSVS--
# salvar df_previsao
df_previsao.to_csv("df_previsao.csv", index=False)

# salvar X_train
X_train.to_csv("X_train.csv", index=False)

# salvar X_test
X_test.to_csv("X_test.csv", index=False)

# salvar y_train
y_train.to_csv("y_train.csv", index=False)

# salvar y_test
y_test.to_csv("y_test.csv", index=False)
