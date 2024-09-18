# House Prices do Kaggle – Tecnicas de Regressão

Neste projeto utilizamos um dataset que possui dados reais de preços de casas para realizar previsões. Além de técnicas de importação e manipulação de dados, são demostradas a plotagem de gráfico, bem como a criação de modelos de regressão com e técnicas de avaliação de precisão.

- Vamos utilizar o dataset disponível no Kaggle. Este é um dataset de competição. 
- Temos uma base com a descrição de cada uma das colunas (data_description.txt)



1) Tecnologias e bibliotecas utilizadas:
![Static Badge](https://img.shields.io/badge/python--green)
![Static Badge](https://img.shields.io/badge/jupyter--green)
![Static Badge](https://img.shields.io/badge/vscode--green)
![Static Badge](https://img.shields.io/badge/pandas--blue)
![Static Badge](https://img.shields.io/badge/matplotlib--blue)
![Static Badge](https://img.shields.io/badge/scikit_learn--orange)


### 1) Iniciando o Projeto
para dar início ao projeto, importaremos o pandas e faremos a leitura da base de treino para um dataframe.

```
# Importando o pandas

import pandas as pd

# Importando o dataset de treino

base = pd.read_csv('train.csv')

# Visualizando essa base

base.head(3)
```
![cover](images/base1.png)

### 2) Visualização da base
Vamos verificar o Shape e as informações da base
```
# Retornando o shape da base

base.shape

(1460, 81)


base.info()

<class 'pandas.core.frame.DataFrame'>

RangeIndex: 1460 entries, 0 to 1459

Data columns (total 81 columns):

 #   Column         Non-Null Count  Dtype 

---  ------         --------------  ----- 

 0   Id             1460 non-null   int64 

 1   MSSubClass     1460 non-null   int64 

 2   MSZoning       1460 non-null   object

 3   LotFrontage    1201 non-null   float64

 4   LotArea        1460 non-null   int64 

 5   Street         1460 non-null   object
...

 78  SaleType       1460 non-null   object

 79  SaleCondition  1460 non-null   object

 80  SalePrice      1460 non-null   int64 

dtypes: float64(3), int64(35), object(43)

memory usage: 924.0+ KB

```
Observe que a nossa base não tem um número tão grande de linhas, mas 81 é um número muito grande de colunas para esta base, pode ser que ocorra overfitting.

### 2) Visualização de dados nulos
```
# Visualizando quantidade de valores vazios

(base.isnull().sum()/base.shape[0]).sort_values(ascending=False).head(20)

PoolQC          0.995205

MiscFeature     0.963014

Alley           0.937671

Fence           0.807534

FireplaceQu     0.472603

LotFrontage     0.177397

GarageYrBlt     0.055479

GarageCond      0.055479

GarageType      0.055479

GarageFinish    0.055479

GarageQual      0.055479

BsmtFinType2    0.026027

BsmtExposure    0.026027

BsmtQual        0.025342

BsmtCond        0.025342

BsmtFinType1    0.025342

MasVnrArea      0.005479

MasVnrType      0.005479

Electrical      0.000685

Id              0.000000

dtype: float64
```
https://www.hashtagtreinamentos.com/house-price-do-kaggle-ciencia-de-dados
https://github.com/IbrahimSobh/kaggle-COVID19-Classification/blob/master/README.md?plain=1
https://ibrahimsobh.github.io/kaggle-COVID19-Classification/
https://www.datacamp.com/tutorial/complete-guide-data-augmentation