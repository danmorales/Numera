import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

dado = pd.read_csv('googleplaystore_user_reviews.csv',encoding="latin1")

print("Exibindo 5 primeiras linhas")
print(dado.head())

print("\n")
print("Combinando as colunas revisão e sentimento num novo arranjo")

dado2=pd.concat([dado.Translated_Review,dado.Sentiment],axis=1)

print("\n")
print("Removendo linhas que estejam faltando informações")
dado2.dropna(axis=0,inplace=True)

print("\n")
print("Exibindo o novo arranjo")
print(dado2.head(10))

print("\n")
print("Verificando os tipos de sentimentos")
print(dado2.Sentiment.unique() )

print("\n")
print("Convertendo sentimentos para valores do tipo inteiro")
print("0: Positivo, 1: Negativo e 2: Neutro")

dado2.Sentiment=[0 if i=="Positive" else 1 if i== "Negative" else 2 for i in dado2.Sentiment]

print("\n")
print("Graficando os sentimentos")

plt.figure(1)
sns.countplot(dado2.Sentiment)
plt.title("Contagem de sentimentos")
plt.xlabel("Sentimento")
plt.ylabel("Contagem")
plt.tight_layout()
plt.show()
savefig('Fig13-Contagem_sentimentos.png',dpi=100)

print("\n")
print("Exibindo as contagens dos sentimentos")
contagem_sentimento = dado2.Sentiment.value_counts()
print(contagem_sentimento)

print("\n")
print("Removendo caracteres que não são letras")
primeira_amostra = dado2.Translated_Review[0]
texto=re.sub("[^a-zA-Z]"," ",primeira_amostra)
texto=texto.lower()

print("Exibindo texto original {} e texto alterado {}".format(primeira_amostra,texto))
print(texto)

print("\n")
print("Aplicando Tokenização")
texto_token=nltk.word_tokenize(texto)
print(texto_token)

print("\n")
print("Aplicando lemmatização")
lemma=nltk.WordNetLemmatizer()
texto_lemma=[lemma.lemmatize(i) for i in texto_token]
texto_lemma=" ".join(texto_lemma)
print(texto_lemma)

print("\n")
print("Aplicando lemmatização no texto inteiro")

lista_textos = []

for i in dado2.Translated_Review:
	texto = re.sub("[^a-zA-Z]"," ",i)
	texto=texto.lower()
	texto=nltk.word_tokenize(texto)
	texto_lemma=nltk.WordNetLemmatizer()
	texto=[texto_lemma.lemmatize(palavra) for palavra in texto]
	texto=" ".join(texto)
	lista_textos.append(texto)
	
print(lista_textos[:10])

max_caracteristicas = 200000

count_vec = CountVectorizer(max_features=max_caracteristicas,stop_words="english")

matrix_esparsa=count_vec.fit_transform(lista_textos).toarray()

todas_palavras = count_vec.get_feature_names()

print("\n")
print("Palavras mais utilizadas")
print(todas_palavras)

plt.figure(2,figsize=(12,12))
nuvem=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(todas_palavras[100:]))
plt.imshow(nuvem)
plt.axis("off")
plt.show()
savefig('Fig14- WordCloud.png',dpi=100)

