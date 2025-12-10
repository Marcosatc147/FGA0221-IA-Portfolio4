# Projeto 1: Aprendizado Supervisionado (Classificação)
# Arquivo: projeto1_supervisionado_classificacao.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Carregamento
df = pd.read_csv('../robot_2.csv')

# 2. Análise Crítica Inicial (Para o relatório)
# "Tentamos prever o preço inicialmente, mas a correlação era inexistente.
# Pivotamos o problema para identificar padrões de engenharia entre as marcas."

# 3. Pré-processamento
# Vamos remover o Preço e o Modelo (que revela a marca)
# Queremos saber se as SPECS (bateria, sensores, sucção) definem a MARCA.
X = df.drop(['Price_USD', 'Brand', 'Model'], axis=1)
y = df['Brand']

# Codificar as variáveis categóricas de entrada (Features)
le_dict = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# 4. Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Modelo (Random Forest Classifier)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Avaliação
y_pred = clf.predict(X_test)

print("Acurácia do Modelo:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

# Matriz de Confusão (Visualização Obrigatória para o Portfólio)
plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title('Matriz de Confusão: A IA consegue distinguir as marcas?')
plt.ylabel('Marca Real')
plt.xlabel('Marca Prevista pelo Modelo')
plt.savefig('projeto1_matriz_confusao.png')
print("Gráfico salvo como 'projeto1_matriz_confusao.png'")

# 7. Análise de Importância das Features (O "Porquê")
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nO que define uma marca? (Top 5 Features):")
print(feature_imp.head(5))