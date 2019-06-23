import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

dado1 = pd.read_csv('googleplaystore.csv')

print(dado1.head())

tamanho = dado1.shape
linhas = tamanho[0]
colunas = tamanho[1]

print("\n")
print("Este arquivo possui {} linhas e {} colunas".format(linhas,colunas))

nomes_colunas = list(dado1.columns)

print("\n")
print("Nomes das colunas")
print(nomes_colunas)

#Determinando se existem dados ausentes
#Calculando a quantidade de dados ausentes de cada variável

total_nans = dado1.isnull().sum().sort_values(ascending=False)
percentual_nans = (dado1.isnull().sum()/dado1.isnull().count()).sort_values(ascending=False)

dados_ausentes = pd.concat([total_nans, percentual_nans], axis=1, keys=['Total', 'Percentual'])

print("\n")
print("Exibindo a quantidade de dados ausentes de cada variável")
print(dados_ausentes)

print("\n")
print("Observa-se que as variáveis Rating, Current Ver, Android Ver, Contente Rating e Type possui valores ausentes")
print("Vamos agora remover as linhas que contenham dados ausentes")

dado1.dropna(how ='any', inplace = True)

#Determinando se ainda existem dados ausentes após a remoção dos mesmos
#Calculando a quantidade de dados ausentes de cada variável

total_nans2 = dado1.isnull().sum().sort_values(ascending=False)
percentual_nans2 = (dado1.isnull().sum()/dado1.isnull().count()).sort_values(ascending=False)

dados_ausentes2 = pd.concat([total_nans2, percentual_nans2], axis=1, keys=['Total', 'Percentual'])

print("\n")
print("Exibindo a quantidade de dados ausentes de cada variável após a remoção")
print(dados_ausentes2)

tamanho2 = dado1.shape
linhas2 = tamanho2[0]
colunas2 = tamanho2[1]

print("\n")
print("Este dado possúi {} linhas e {} colunas após a remoção dos dados ausentes".format(linhas2,colunas2))

print("\n")
print("Exibindo os detalhes das notas de cada aplicativo")
detalhe_notas = dado1['Rating'].describe()
print(detalhe_notas)

print("\n")
print("Gerando gráfico da distribuição das notas")

plt.figure(1)
sns.kdeplot(dado1.Rating, color="Blue", shade = True,label='Notas',legend=False)
plt.xlabel("Notas")
plt.ylabel("Frequencia")
plt.title("Distribuição das notas")
plt.tight_layout()
plt.show()
savefig('Fig1-Distribuicao_notas.png',dpi=100)

print("\n")
print("Determinando o número de categorias e exibindo o nome delas")
categorias = dado1['Category'].unique()
num_categorias = len(categorias)

print("Número de categorias = {}".format(num_categorias))
print("Categorias")
print(categorias)

print("\n")
print("Graficando o número de aplicativos por categoria")

plt.figure(2,figsize=(15,7))
ax = sns.countplot(x="Category",data=dado1, palette = "Set1")
ax.set_xlabel("Categoria")
ax.set_ylabel("Contagem")
ax.set_title('Contagem de aplicativos de cada categoria')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
plt.tight_layout()
plt.show()
savefig('Fig2-Contagem_de_aplicativos_categoria.png',dpi=100)

print("\n")
print("Observamos que as categorias Games e Familia possuem maior contagem")

print("\n")
print("Graficando as notas em funções das categorias num gráfico boxplot")
ax2 = sns.catplot(x="Category",y="Rating",data=dado1, kind="box", height = 7 , palette = "Set1",aspect=1.2)
ax2.set_xticklabels(rotation=90)
ax2.set_xlabels("Categorias")
ax2.set_ylabels("Notas")
plt.tight_layout()
plt.show()
savefig('Fig3-Boxplot_Notas_Categorias.png',dpi=100)

print("\n")
print("Verificando o número de vezes que cada aplicativo foi instalado")
print(dado1['Installs'].head())

print("\n")
print("Removendo o sinal de mais, vírgula e convertendo para número inteiro os dados de quantas vezes foi instalado")
dado1.Installs = dado1.Installs.apply(lambda x: x.replace(',',''))
dado1.Installs = dado1.Installs.apply(lambda x: x.replace('+',''))
dado1.Installs = dado1.Installs.apply(lambda x: int(x))

print("\n")
print("Verificando novamente o número de vezes que cada aplicativo foi instalado")
print(dado1['Installs'].head())

print("\n")
print("Ordenando o número de instalações")
dado_install_sort = sorted(list(dado1['Installs'].unique()))
dado1['Installs'].replace(dado_install_sort,range(0,len(dado_install_sort),1), inplace=True )

print("\n")
print("Graficando as notas em função do número de vezes que foi instalado")
plt.figure(4,figsize = (10,10))
plt.scatter(dado1.Installs,dado1.Rating,color='red')
plt.xlabel("Número de instalações")
plt.ylabel("Nota")
plt.tight_layout()
plt.show()
savefig('Fig4-Install_vs_nota.png',dpi=100)

print("\n")
print("Exibindo valores da coluna Type")
print(dado1['Type'].unique())

print("\n")
print("Graficando número de aplicativos pagos e gratuitos")

plt.figure(5,figsize = (6,6))
ax3 = sns.countplot(dado1.Type)
ax3.set_title("Gratuito vs Pago")
ax3.set_xlabel("Tipo")
ax3.set_ylabel("Contagem")
plt.tight_layout()
plt.show()
savefig('Fig5-CountPlot_tipo_vs_contagem.png',dpi=100)

print("\n")
print("Verificando correlação dos dados e graficando")
plt.figure(6,figsize = (6,6))
correlacao = dado1.corr()
cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(correlacao, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
plt.tight_layout()
plt.show()
savefig('Fig6-heatmap_correlacao.png',dpi=100)

print("\n")
print("Graficando detalhes das cinco categorias mais caras")

dado1.Price = dado1.Price.str.replace("$","")
dado1.Price = dado1.Price.astype('float')

dado1_ordenado=dado1.sort_values(by='Price',ascending=False)

preco_alto = dado1_ordenado.Price[0:5]
label_alto = dado1_ordenado.Category[0:5]

ax3 = sns.catplot(x="Content Rating",y="Rating",data=dado1, kind="box", height=6 ,palette = "Paired")
ax3.despine(left=True)
ax3.set_xticklabels(rotation=90)
ax3 = ax3.set_ylabels("Notas")
plt.title('Box plot das notas em função da faixa etária')
savefig('Fig7-BoxPlot-Nota-FaixaEtaria.png',dpi=100)

print("\n")
print("Classificando por faixa etária")

todos = []
adolescentes = []
acima10 = []
acima17 = []
adultos = []
indefinido = []

todos.append(len(dado1[(dado1['Content Rating']=="Everyone")]))
adolescentes.append(len(dado1[(dado1['Content Rating']=="Teen")]))
acima10.append(len(dado1[(dado1['Content Rating']=="Everyone 10+")]))
acima17.append(len(dado1[(dado1['Content Rating']=="Mature 17+")]))
adultos.append(len(dado1[(dado1['Content Rating']=="Adults only 18+")]))
indefinido.append(len(dado1[(dado1['Content Rating']=="Unrated")]))

yy = [todos,adolescentes,acima10,acima17,adultos,indefinido]

plt.figure(8,figsize=(8,8))
plt.plot(dado1['Content Rating'].unique(),yy)
plt.xlabel('Faixa etária')
plt.ylabel('Quantidade de instalações')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
savefig('Fig8-Faixa_etaria.png',dpi=100)

print("\n")
print("Graficando a nota média de cada categoria")
x_nome_categoria = dado1['Category'].unique()
y_nota_categoria = dado1.groupby('Category')['Rating'].mean().values

plt.figure(9,figsize=(8,6))
sns.barplot(x=x_nome_categoria,y=y_nota_categoria)
plt.xticks(rotation=90)
plt.xlabel('Categoria')
plt.ylabel('Nota média')
plt.tight_layout()
plt.show()
savefig('Fig9-barplot_categoria_vs_nota.png',dpi=100)

print("\n")
print("Convertendo Reviews de string para float")
dado1.Reviews=dado1.Reviews.astype(float)
numero_reviews = dado1.groupby('Category')['Reviews'].sum().values

plt.figure(10,figsize=(8,6))
plt.scatter(x_nome_categoria,numero_reviews)
plt.xlabel('Categoria')
plt.ylabel('Número de revisões')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
savefig('Fig10-categoria_vs_revisao.png',dpi=100)

generos = dado1.Genres.unique()
contagem_generos = dado1.Genres.value_counts()

plt.figure(11,figsize=(10,10))
sns.barplot(x=contagem_generos.index[:20],y=contagem_generos[:20])
plt.xlabel('Genero')
plt.ylabel('Contagem')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
savefig('Fig11-genero_vs_contagem.png',dpi=100)

labels=["Gratuito","Pago"]

p_tools = dado1[dado1['Genres']=='Tools'].Type.value_counts().values
p_entre = dado1[dado1['Genres']=='Entertainment'].Type.value_counts().values
p_education = dado1[dado1['Genres']=='Education'].Type.value_counts().values

label_tools = dado1[dado1['Genres']=='Tools'].Type.value_counts().index
label_entre = dado1[dado1['Genres']=='Entertainment'].Type.value_counts().index
label_education = dado1[dado1['Genres']=='Education'].Type.value_counts().index

plt.figure(12,figsize=(8,4))
plt.subplot(1, 3, 1)
plt.pie(p_tools,labels=label_tools, autopct='%1.1f%%')
plt.title("Tipo Tools")
plt.subplot(1, 3, 2)
plt.pie(p_entre,labels=label_entre, autopct='%1.1f%%')
plt.title("Tipo Entertainment")
plt.subplot(1, 3, 3)
plt.pie(p_education,labels=label_education, autopct='%1.1f%%')
plt.title("Tipo Education")
plt.tight_layout()
plt.show()
savefig('Fig12-pieplot_tipos.png',dpi=100)