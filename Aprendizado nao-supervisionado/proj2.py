import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

print("Iniciando Clusterização...")

# 1. Carregamento
df = pd.read_csv('../robot_2.csv')

# 2. Seleção de Features (Apenas técnicas)
features_tecnicas = [
    'Battery_Life_Minutes', 'Charging_Time_Hours', 'Dustbin_Capacity_ml', 
    'Suction_Power_Pa', 'Noise_Level_dB', 'Number_of_Sensors',
    'Weight_kg', 'Height_cm'
]
# Features binárias que importam
cols_bin = ['WiFi_Connectivity', 'App_Control', 'Obstacle_Avoidance', 
            'HEPA_Filter', 'Auto_Dock', 'Mop_Function', 'Self_Emptying']

X = df[features_tecnicas + cols_bin].copy()

# Encodar binárias
le = LabelEncoder()
for col in cols_bin:
    X[col] = le.fit_transform(X[col])

# 3. Normalização (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. K-Means
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# 5. Visualização (PCA 2D)
print("Gerando gráfico PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=80, alpha=0.8)
plt.title('Segmentação de Mercado (K-Means + PCA)')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Cluster')
plt.grid(True, alpha=0.3)
plt.savefig('proj2_clusters.png')
print("Gráfico salvo: proj2_clusters.png")

# 6. Análise dos Grupos (Print no terminal para você ler)
print("\n=== PERFIL DOS GRUPOS ENCONTRADOS ===")
print(df.groupby('Cluster')[['Suction_Power_Pa', 'Battery_Life_Minutes', 'Price_USD']].mean())