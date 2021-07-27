#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Desativando os avisos sobre as bibliotecas depreciadas:
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Importando as bibliotecas 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import collections

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# In[2]:


# Lendo o conjunto de dados
cliente_info = pd.read_csv('application_record.csv')
status_credito = pd.read_csv('credit_record.csv')


# # Pré-processamento dos Dados - Cliente_info

# Vamos verificar o número de linhas do nosso dataframe e quais os tipos dos dados

# In[3]:


# Informações do dataframe
cliente_info.info()


# In[4]:


cliente_info.head()


# In[5]:


# Mudando o nome das colunas do dataframe para melhor entendimento
cliente_info.rename(columns={'CODE_GENDER':'Genero','FLAG_OWN_CAR':'Carro','FLAG_OWN_REALTY':'Propriedade',
                             'CNT_CHILDREN':'Filhos','AMT_INCOME_TOTAL':'Ganho_Anual',
                             'NAME_EDUCATION_TYPE':'Estudo','NAME_FAMILY_STATUS':'Estado_Civil',
                             'NAME_HOUSING_TYPE':'Moradia','FLAG_EMAIL':'email',
                             'NAME_INCOME_TYPE':'Empregado','FLAG_WORK_PHONE':'Telefone_Trabalho',
                             'FLAG_PHONE':'Telefone','CNT_FAM_MEMBERS':'Membros Familia',
                             'OCCUPATION_TYPE':'Ocupacao', 'DAYS_BIRTH':'Idade',
                             'DAYS_EMPLOYED':'Dias_Trabalhados', 'FLAG_MOBIL':'Celular'
                            },inplace=True)


# In[6]:


# primeiras linhas do Dataframe
cliente_info.head()


# In[7]:


# Verificar valores estatísticos 
cliente_info.describe()


# Veficando os ids unicos e compartilhados 

# In[8]:


print(len(set(cliente_info['ID']))) # Quantos IDs unicos a tabela cliente_info possui?
print(len(set(status_credito['ID']))) # Quantos IDs unicos a tabela status_cliente possui?
print(len(set(cliente_info['ID']).intersection(set(status_credito['ID'])))) # Quantos IDs as duas tabelas compartilham?


# In[9]:


# Checando valores faltantes
cliente_info.isnull().sum()


# A coluna Ocupacao por ter muitos valores faltantes e ter uma perda alta de 134203 registros, optaremos por deletar

# In[10]:


# Ocupacao: Muitos valores faltantes
cliente_info.drop(columns=['Ocupacao'], inplace=True)


# Vamos converter todos os valores categóricos em valores numéricos. A Variáveis dummy são variáveis binárias (0 ou 1) criadas para representar uma variável com duas ou mais categorias.

# In[11]:


# Criando váriaveis dummy
cliente_info = pd.get_dummies(cliente_info, columns = ["Genero", "Carro", "Propriedade", "Estudo",
                                                       "Moradia", "Empregado", "Estado_Civil"], drop_first=True)


# Substituir valores que estão errados na coluna Dias Trabalhados por um valor neutro, evitando apagar e perder dados se o cliente tem valor >= 0 então ele não trabalhou

# In[12]:


# Organizando a coluna Dias_Trabalhados para quem não trabalhou para o valor 0
cliente_info["Dias_Trabalhados"] = np.where(cliente_info["Dias_Trabalhados"] > 0, 0, cliente_info["Dias_Trabalhados"])


# Procurando colunas com valores duplicados. Um único cliente com vários ids

# In[13]:


# Procurando valores duplicados
cliente_info.loc[cliente_info.Idade== -18926].loc[cliente_info.Dias_Trabalhados==-6276]


# Apagando os valores duplicados e verificando se ainda existe duplicação

# In[14]:


# Apagando e verificando 
cliente_info = cliente_info.drop_duplicates(subset=cliente_info.columns[1:], keep='first', inplace=False)
cliente_info.loc[cliente_info.Idade== -18926]


# Podemos notar que se dividirmos o numero da coluna Aniversario por 365, obteremos a real idade do cliente

# In[15]:


# corrigindo a idade e observando os calculos estatísticos
cliente_info['Idade']=-(cliente_info['Idade'])//365


# In[16]:


# Gráfico das idades dos clientes e da quantidade de clientes por idade
plt.style.use('fivethirtyeight')

plt.hist(cliente_info['Idade'], edgecolor = 'k', bins = 25)
plt.title('Idade do Cliente'); plt.xlabel('Idade'); plt.ylabel('Quantidade');


# # Pré-processamento dos Dados - status_credito

# In[17]:


# Informações do dataframe:
status_credito.info()


# In[18]:


# primeiras linhas do Dataframe
status_credito.head()


# In[19]:


# Verrificando se possui valores faltantes
status_credito.isnull().sum()


# In[20]:


# A coluna MONTHS_BALANCE parece confusa e não possui uma influência real nos dados
status_credito.drop('MONTHS_BALANCE', axis=1, inplace=True)


# Situação do Credito:
# 
# X, C e 0 bons pagadores /// 1, 2, 3, 4 e 5 devedores 

# In[21]:


# Quantidade de vezes de cada avaliação na coluna status:
repetidos = status_credito['STATUS']
collections.Counter(repetidos)


# Padronizar X,C e 0 como 0 e (1,2,3,4,5) como 1

# In[22]:


status_credito.STATUS.replace('X', 0, inplace=True)
status_credito.STATUS.replace('C', 0, inplace=True)
status_credito.STATUS = status_credito.STATUS.astype('int') # Transformar a coluna status em inteiro
status_credito.STATUS.replace(2, 1, inplace=True)
status_credito.STATUS.replace(3, 1, inplace=True)
status_credito.STATUS.replace(4, 1, inplace=True)
status_credito.STATUS.replace(5, 1, inplace=True)


# Verificando a quantidade e a porcentagem de cada valor da classe

# In[23]:


# Quantidade de aprovados 1 e não aprovados 0 (em números e porcentagem)
print(status_credito['STATUS'].value_counts())
status_credito['STATUS'].value_counts(normalize=True)


# In[24]:


# Gráfico de créditos aprovados e desaprovados
sns.countplot(x='STATUS', data=status_credito, palette='CMRmap')
print('Aprovados: {}%'.format(round(status_credito.STATUS.value_counts()[0]/len(status_credito)*100.0,2)))
print('Desaprovados: {}%'.format(round(status_credito.STATUS.value_counts()[1]/len(status_credito)*100.0,2)))


# # Unindo os dois Datasets:

# In[25]:


# Unindo os 2 datasets
uniao = pd.merge(cliente_info,status_credito, on='ID', how='left')
uniao.dropna(inplace=True)


# In[26]:


# primeiras linhas do Dataframe
uniao.info()


# Visualizando cada uma das colunas

# In[27]:


# Visualização
uniao.hist(figsize=(35, 20));


# In[28]:


# Mapa de calor para encontrar correlações
plt.figure(figsize=(10,5))
sns.heatmap(data=uniao.corr(), cmap="seismic")
plt.show();


# As mais altas correlações:
# - Filhos x Membros Familia
# - Estudo_Higher education x Estudo_Secondary / secondary special
# 
# Apesar dessas correlações serem fortes, não espero que ocorra algum risco de multicolinearidade. 

# # Escalonamento

# Alguns valores irão precisar de escalonamento como: Ganho_Anual e Dias_Trabalhados. Assim esses valores vão estar com um intervalo menor do maior até o menor

# In[29]:


# Escalonamento de Ganho_Anual e Dias_Trabalhados
escalonamento = ['Ganho_Anual','Dias_Trabalhados']
scaler = preprocessing.RobustScaler()
uniao[['Ganho_Anual', 'Dias_Trabalhados']] = scaler.fit_transform(uniao[['Ganho_Anual', 'Dias_Trabalhados']])


# In[30]:


# Como podemos verificar os valores foram escalonados
uniao[['Dias_Trabalhados','Ganho_Anual']].describe()


# In[ ]:





# Aqui vamos separar a classe dos previsores para podermos criar os modelos

# In[31]:


# Definição da classe e dos previsores
classe = uniao['STATUS']
previsores = uniao[['Ganho_Anual', 'Idade',
       'Dias_Trabalhados', 'Membros Familia', 'Genero_M', 'Carro_Y',
       'Propriedade_Y', 'Estudo_Higher education', 'Estudo_Incomplete higher', 'Moradia_Rented apartment',
       'Estudo_Lower secondary',  'Moradia_House / apartment','Filhos','Estudo_Secondary / secondary special',
       'Moradia_Municipal apartment', 'Moradia_Office apartment',  
       'Moradia_With parents', 'Empregado_Pensioner', 'Empregado_State servant', 'Empregado_Student',
       'Empregado_Working', 'Estado_Civil_Married',  'Telefone_Trabalho', 'email',
       'Estado_Civil_Separated', 'Estado_Civil_Single / not married', 'Telefone', 'Celular',
       'Estado_Civil_Widow']]


# As técnicas de Machine Learning têm uma tendência para a classe majoritária e tendem a ignorar a classe minoritária. Eles tendem apenas a prever a classe majoritária, portanto, tendo uma grande classificação incorreta da classe minoritária em comparação com a classe majoritária. Em palavras mais técnicas, se tivermos distribuição de dados desequilibrada em nosso conjunto de dados, então nosso modelo se torna mais sujeito ao caso em que a classe minoritária tem memória insignificante ou muito menor.

# In[32]:


# Usaremos SMOTE, que sintetiza novas instâncias minoritárias entre instâncias minoritárias existentes.
smt = SMOTE()
previsores,classe = smt.fit_sample(previsores,classe)


# In[33]:


# Agora o problema de desbalanceamento está resolvido como mostra o gráfico abaixo
sns.countplot(x=classe, data=uniao, palette='CMRmap')


# Agora, dividiremos nossos dados em conjunto de treinamento e conjunto de teste para preparar nossos dados para duas fases diferentes de modelagem de aprendizado de máquina: treinamento e teste. O ideal é que nenhuma informação dos dados de teste seja usada para dimensionar os dados de treinamento ou para direcionar o processo de treinamento de um modelo de aprendizado de máquina. Portanto, primeiro dividimos os dados e, em seguida, aplicamos a escala.

# In[34]:


# Dividindo os dados de teste e treinamento para os modelos
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe,
                                                                                    test_size=0.25, random_state=0)


# # MACHINE LEARNING

# Agora avaliaremos nosso modelo no conjunto de teste com relação à precisão da classificação. Mas também daremos uma olhada na matriz de confusão do modelo. No caso de previsão de aplicativos de cartão de crédito, é igualmente importante ver se nosso modelo de aprendizado de máquina é capaz de prever o status de aprovação dos aplicativos como negados e originalmente negados. Se nosso modelo não tiver um bom desempenho nesse aspecto, ele pode acabar aprovando o que não deveria ter sido aprovado. A matriz de confusão nos ajuda a ver o desempenho de nosso modelo a partir desses aspectos.

# ARVORE DE DECISÃO

# In[35]:


classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# In[36]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_ad = accuracy_score(classe_teste, previsoes)
print('A precisão de acerto é:', precisao_ad)
class_names = ['aprovados', 'desaprovados']
matrix = confusion_matrix(classe_teste, previsoes)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Matriz de Confusão"), plt.tight_layout()
plt.show()


# In[37]:


print (classification_report(classe_teste,previsoes))


# In[38]:


# Importância dos Recursos
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
tmp = pd.DataFrame({'Recursos': previsores, 'Importância dos Recursos': classificador.feature_importances_})
tmp = tmp.sort_values(by='Importância dos Recursos',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Importância dos recursos',fontsize=14)
s = sns.barplot(x='Recursos',y='Importância dos Recursos',data=tmp,lw=1.5)
s.set_xticklabels(previsores,rotation=90)
plt.show()


# NAIVE BAYES

# In[39]:


# Algoritmo NAIVE BAYES criado
classificador = GaussianNB()
classificador.fit(previsores_treinamento,classe_treinamento)
# previsões do algoritmo NAIVE BAYES 
previsoes = classificador.predict(previsores_teste)


# In[40]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_nb = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste,previsoes)
print('A precisão de acerto é:', precisao_nb)
print('A matriz de confusão é:')
print(matriz)


# In[41]:


print (classification_report(classe_teste,previsoes))


# In[ ]:





# REGRESSÃO

# In[42]:


classificador = LogisticRegression(C=0.2,
                           random_state=0,
                           solver='liblinear')
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# In[43]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_rl = accuracy_score(classe_teste, previsoes)
print('A precisão de acerto é:', precisao_rl)
class_names = ['aprovados', 'desaprovados']
matrix = confusion_matrix(classe_teste, previsoes)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Matriz de Confusão"), plt.tight_layout()
plt.show()


# In[44]:


print (classification_report(classe_teste,previsoes))


# In[ ]:





# RANDOM FOREST

# In[45]:


classificador = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# In[46]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_rf = accuracy_score(classe_teste, previsoes)
print('A precisão de acerto é:', precisao_rf)
class_names = ['aprovados', 'desaprovados']
matrix = confusion_matrix(classe_teste, previsoes)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Matriz de Confusão"), plt.tight_layout()
plt.show()


# In[47]:


print (classification_report(classe_teste,previsoes))


# In[48]:


# Importância dos Recursos
sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
tmp = pd.DataFrame({'Recursos': previsores, 'Importância dos Recursos': classificador.feature_importances_})
tmp = tmp.sort_values(by='Importância dos Recursos',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Importância dos recursos',fontsize=14)
s = sns.barplot(x='Recursos',y='Importância dos Recursos',data=tmp,lw=1.5)
s.set_xticklabels(previsores,rotation=90)
plt.show()   


# CATBOOST

# In[49]:


classificador = CatBoostClassifier()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# In[50]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_cat = accuracy_score(classe_teste, previsoes)
print('A precisão de acerto é:', precisao_cat)
class_names = ['aprovados', 'desaprovados']
matrix = confusion_matrix(classe_teste, previsoes)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Matriz de Confusão"), plt.tight_layout()
plt.show()


# In[51]:


print (classification_report(classe_teste,previsoes))


# In[52]:


# Importância dos Recursos
tmp = pd.DataFrame({'Recursos': previsores, 'Importância dos Recursos': classificador.feature_importances_})
tmp = tmp.sort_values(by='Importância dos Recursos',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Importância dos recursos',fontsize=14)
s = sns.barplot(x='Recursos',y='Importância dos Recursos',data=tmp,lw=1.5)
s.set_xticklabels(previsores,rotation=90)
plt.show()   


# XGBOOST

# In[53]:


classificador = XGBClassifier()
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# In[54]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_xgb = accuracy_score(classe_teste, previsoes)
print('A precisão de acerto é:', precisao_xgb)
class_names = ['aprovados', 'desaprovados']
matrix = confusion_matrix(classe_teste, previsoes)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Matriz de Confusão"), plt.tight_layout()
plt.show()


# In[55]:


print (classification_report(classe_teste,previsoes))


# In[56]:


# Importância dos Recursos
tmp = pd.DataFrame({'Recursos': previsores, 'Importância dos Recursos': classificador.feature_importances_})
tmp = tmp.sort_values(by='Importância dos Recursos',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Importância dos recursos',fontsize=14)
s = sns.barplot(x='Recursos',y='Importância dos Recursos',data=tmp,lw=1.5)
s.set_xticklabels(previsores,rotation=90)
plt.show()   


# LIGHTGBM

# In[57]:


classificador = LGBMClassifier(num_leaves=31,
                       max_depth=8, 
                       learning_rate=0.02,
                       n_estimators=250,
                       subsample = 0.8,
                       colsample_bytree =0.8
                      )
classificador.fit(previsores_treinamento, classe_treinamento)
previsoes = classificador.predict(previsores_teste)


# In[58]:


#Porcentagem de acerto e a matriz de decisão(placar de erros e acertos)                                                                                             
precisao_lgbm = accuracy_score(classe_teste, previsoes)
print('A precisão de acerto é:', precisao_lgbm)
class_names = ['aprovados', 'desaprovados']
matrix = confusion_matrix(classe_teste, previsoes)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Matriz de Confusão"), plt.tight_layout()
plt.show()


# In[59]:


print (classification_report(classe_teste,previsoes))


# In[60]:


# Importância dos Recursos
tmp = pd.DataFrame({'Recursos': previsores, 'Importância dos Recursos': classificador.feature_importances_})
tmp = tmp.sort_values(by='Importância dos Recursos',ascending=False)
plt.figure(figsize = (7,4))
plt.title('Importância dos recursos',fontsize=14)
s = sns.barplot(x='Recursos',y='Importância dos Recursos',data=tmp,lw=1.5)
s.set_xticklabels(previsores,rotation=90)
plt.show()   


# In[61]:


y = np.array([precisao_ad,precisao_rf,precisao_rl,precisao_nb,precisao_cat,precisao_xgb, precisao_lgbm ])
x = ["ArvoreDecisao","RandomForest","RegressaoLogistica","NaiveBayes","CatBoost","XGBoost","LightGBM"]
plt.barh(x,y)
plt.title("Comparação dos Modelos")
plt.xlabel("Precisao")
plt.show()


# In[62]:


print('Random Forest: ',precisao_rf)
print('Árvore de Decisão: ',precisao_ad)
print('CatBoost: ',precisao_cat)
print('XGBoost: ',precisao_xgb)
print('LightGbm: ',precisao_lgbm)
print('Regressção Logística: ',precisao_rl)
print('Naive Bayes: ',precisao_nb)


# In[ ]:




