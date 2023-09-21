# Sales Forecasting Experiment
Aqui temos um modelo da previsão da quantidade vendidas de itens usando
Python e scikit-learn. Depois de explorar dados e diversos resultados
a opção foi por um modelo Random Forest que pode ser usado para a previsão das
vendas de cada item para os ciclos 202016, 202017 e 202101.

**Nota:** É importante ressaltar que em um caso real para evitar data leaking
e realizar o alinhamento aos preceitos de Time Series a separção de train
test e validation seria feito como maior ou menor que a data X.
Nesse caso como queria experimentar e aplicar conceitos específicos utilizei
uma divisão randômica.

## Estrutura dos arquivos
- `analise_exploratoria.ipynb`: Jupyter notebook contendo toda a análise
exploratória realizada
- `case_ds_gdem.aqlite3`: Banco de dados SQLite contendo a tabela utilizada
para as análises e construção do modelo. O mesmo foi utilizado no Jupyter
notebook e na etapa de pré processamento
- `pre_processing.py`: arquivo que realiza todo o pre processamento dos
dados e salva os seguintes arquivos csv a serem utilizados para treino
e previsão:
    - `df_previsao.csv`: contém os dados a serem utilizados para
    a previsao de vendas nos ciclos 202016, 202017 e 202101
    - `X_train.csv` e `y_train.csv`: contém os dados utilizados para treino
    - `X_test.csv` e `y_test.csv`: contém os dados utilizados para testar
    as previsões do modelo
- `treino_modelo.py`: Arquivo que realiza o treino a avaliacao da performance
do Random Forest Regressivo. Aqui também salvamos o modelo treinado em
um arquivo pickle (`random_forest_regressor.pkl`)
- `modelo_previsao.py`: Arquivo que realiza a preevisao do modelo
- `previsao_gerada_modelo.xlsx`: Contém a previsão gerada pelo modelo

## Como executar a previsão
Como foi dado os ciclos a serem executadas as previsões os dados já foram
tratados e salvos no arquivo `df_previsao.csv`.
O modelo treinado também havida sifo salvo em um arquivo `.pkl`. Portanto para
rodar as previsões somente era necessário realizar o seguinte:
- Rodar o arquivo `modelo_previsao.py`, o mesmo ira salvar a previsão do
modelo em um `.xlsx`
Porém como o arquivo é muito grande ele não se encontra no repositório.
