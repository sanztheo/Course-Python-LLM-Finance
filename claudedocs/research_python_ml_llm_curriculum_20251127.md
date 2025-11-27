# Parcours Python Optimal: Machine Learning & Développement LLM
$$ aa $$
**Date**: 2025-11-27
**Objectif**: Curriculum structuré du niveau fondamental au développement LLM avancé

---

## 📊 Vue d'ensemble du parcours

**Durée totale estimée**: 6-12 mois (selon investissement temps)
- Fondamentaux Python: 3-4 semaines
- Python intermédiaire: 2-3 semaines
- Écosystème Data Science: 4-6 semaines
- Machine Learning: 6-8 semaines
- Deep Learning & PyTorch: 8-12 semaines
- Développement LLM: 8-12 semaines

---

## 🎯 Phase 1: Python Fondamental (3-4 semaines)

### Objectifs d'apprentissage
Construire une base solide en programmation Python avant d'aborder le ML/AI. La maîtrise des fondamentaux est cruciale car les bibliothèques de data science s'appuient sur ces concepts.

### Contenu du module

**Semaine 1: Syntaxe & Types de base**
- Variables et types de données (int, float, str, bool)
- Opérateurs (arithmétiques, comparaison, logiques)
- Entrée/sortie utilisateur (input, print)
- Conversion de types (casting)

**Semaine 2: Structures de contrôle**
- Conditions (if/elif/else)
- Boucles (for, while)
- Instructions break, continue, pass
- Gestion d'erreurs basique (try/except)

**Semaine 3: Structures de données**
- Listes (création, indexation, slicing, méthodes)
- Tuples (immutabilité, unpacking)
- Dictionnaires (clés-valeurs, méthodes)
- Sets (opérations ensemblistes)

**Semaine 4: Fonctions & Modules**
- Définition de fonctions (def, return)
- Paramètres (positionnels, nommés, *args, **kwargs)
- Portée des variables (scope)
- Import de modules (import, from...import)
- Modules standards utiles (math, random, datetime)

### Exercices pratiques

**Niveau débutant:**
1. **Calculatrice**: Créer une calculatrice interactive avec opérations de base
2. **Analyse de texte**: Compter mots, caractères, phrases dans un texte
3. **Gestion de liste**: Système de tâches (ajouter, supprimer, afficher)
4. **Jeu de devinette**: Deviner un nombre avec indices

**Niveau intermédiaire:**
1. **Statistiques descriptives**: Calculer moyenne, médiane, écart-type d'une liste
2. **Manipulation de dictionnaires**: Système d'inventaire avec opérations CRUD
3. **Analyse de données CSV**: Lire et analyser un fichier CSV simple
4. **Générateur de mots de passe**: Avec contraintes de complexité

### Ressources recommandées
- [learnpython.org](https://learnpython.org) - Tutoriels interactifs gratuits
- [freeCodeCamp Python Course](https://www.freecodecamp.org/news/data-science-learning-roadmap/) - Cours complet
- [Kaggle Python](https://www.kaggle.com/learn/python) - Notebooks interactifs

### Critères de validation
✅ Capacité à écrire des fonctions réutilisables
✅ Maîtrise des structures de données (listes, dicts)
✅ Compréhension des boucles et conditions
✅ Gestion basique des erreurs

---

## 🔧 Phase 2: Python Intermédiaire (2-3 semaines)

### Objectifs d'apprentissage
Maîtriser les constructions Python avancées qui permettent d'écrire du code plus efficace, lisible et maintenable - essentielles pour le code de production en ML.

### Contenu du module

**Semaine 1: Programmation fonctionnelle**
- List comprehensions `[x**2 for x in range(10)]`
- Dict & Set comprehensions
- Fonctions lambda
- Map, filter, reduce
- Generators et expressions generator `(x for x in range(1000000))`
- Yield vs return

**Semaine 2: Concepts avancés**
- Decorators (fonction wrapping, @decorator)
  - Mesure de temps d'exécution
  - Logging automatique
  - Cache de résultats (memoization)
- Context managers (with statement)
- Itérateurs et protocole d'itération
- Programmation orientée objet (classes, héritage)

**Semaine 3: Bonnes pratiques**
- Type hints et annotations
- Documentation (docstrings)
- Gestion d'erreurs avancée
- Patterns de code propre (DRY, KISS)

### Exercices pratiques

**List Comprehensions:**
```python
# Exercice 1: Filtrer nombres pairs et mettre au carré
numbers = range(20)
result = [x**2 for x in numbers if x % 2 == 0]

# Exercice 2: Créer dictionnaire de fréquences
text = "hello world"
freq = {char: text.count(char) for char in set(text)}
```

**Generators:**
```python
# Exercice 3: Generator pour séquence Fibonacci
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Exercice 4: Pipeline de traitement de données
def process_large_file(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip().split(',')
```

**Decorators:**
```python
# Exercice 5: Decorator de timing
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__}: {time.time()-start:.4f}s")
        return result
    return wrapper

# Exercice 6: Decorator de cache
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper
```

**Context Managers:**
```python
# Exercice 7: Context manager pour chronométrage
from contextlib import contextmanager
@contextmanager
def timer_context(name):
    start = time.time()
    yield
    print(f"{name}: {time.time()-start:.4f}s")

# Utilisation
with timer_context("Data loading"):
    # Code à chronométrer
    pass
```

### Ressources recommandées
- [Real Python - Decorators](https://realpython.com/primer-on-python-decorators/) - Guide complet sur les decorators
- [Intermediate Python](https://book.pythontips.com/en/latest/) - Livre gratuit en ligne
- [Scientific Python Lectures](https://lectures.scientific-python.org/advanced/advanced_python/index.html) - Constructions avancées

### Critères de validation
✅ Écriture de list/dict comprehensions complexes
✅ Création et utilisation de generators
✅ Implémentation de decorators personnalisés
✅ Compréhension des context managers

---

## 📊 Phase 3: Écosystème Data Science (4-6 semaines)

### Objectifs d'apprentissage
Maîtriser les outils fondamentaux de manipulation et visualisation de données. Ces bibliothèques sont la base de tout workflow ML.

### 3.1 NumPy (1-2 semaines)

**Concepts clés:**
- Arrays NumPy vs listes Python
- Création d'arrays (np.array, np.zeros, np.ones, np.arange, np.linspace)
- Indexation et slicing multi-dimensionnel
- Broadcasting
- Opérations vectorisées (10-100x plus rapides que boucles Python)
- Fonctions mathématiques (np.sum, np.mean, np.std, np.dot)
- Algèbre linéaire (multiplication matricielle, eigenvalues)
- Génération de nombres aléatoires

**Exercices pratiques:**

```python
# Exercice 1: Statistiques basiques
data = np.random.randn(1000)
print(f"Mean: {np.mean(data)}, Std: {np.std(data)}")

# Exercice 2: Normalisation de données
def normalize(arr):
    return (arr - np.mean(arr)) / np.std(arr)

# Exercice 3: Distance entre vecteurs (important pour ML)
def euclidean_distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2)**2))

# Exercice 4: Opérations matricielles
A = np.random.randn(100, 50)
B = np.random.randn(50, 30)
C = np.dot(A, B)  # Multiplication matricielle

# Exercice 5: Broadcasting avancé
# Normaliser chaque colonne d'une matrice
matrix = np.random.randn(1000, 10)
normalized = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)

# Exercice 6: Simuler un réseau de neurones simple
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.random.randn(100, 5)  # 100 samples, 5 features
W = np.random.randn(5, 3)    # weights
output = sigmoid(np.dot(X, W))
```

**Pourquoi crucial pour ML:**
- Manipulation efficace de grandes matrices de données
- Opérations vectorisées essentielles pour performance
- Fondation pour PyTorch et TensorFlow
- Algèbre linéaire utilisée partout en ML

### 3.2 Pandas (2-3 semaines)

**Concepts clés:**
- DataFrames et Series
- Lecture/écriture de données (CSV, Excel, JSON, SQL)
- Indexation (loc, iloc, boolean indexing)
- Manipulation de données (merge, join, concat, groupby)
- Nettoyage de données (valeurs manquantes, duplicatas)
- Transformation de données (apply, map, applymap)
- Agrégations et statistiques groupées

**Exercices pratiques:**

```python
import pandas as pd

# Exercice 1: Chargement et exploration
df = pd.read_csv('data.csv')
print(df.head())
print(df.info())
print(df.describe())

# Exercice 2: Nettoyage de données
# Gérer valeurs manquantes
df.dropna()  # Supprimer lignes avec NaN
df.fillna(df.mean())  # Remplir avec moyenne
df.interpolate()  # Interpolation

# Exercice 3: Transformation de données
# One-hot encoding pour variables catégorielles
df_encoded = pd.get_dummies(df, columns=['category'])

# Exercice 4: Agrégations groupées
# Statistiques par groupe
grouped = df.groupby('category').agg({
    'value': ['mean', 'std', 'count'],
    'price': ['min', 'max']
})

# Exercice 5: Feature engineering
# Créer nouvelles features
df['price_per_unit'] = df['total_price'] / df['quantity']
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Exercice 6: Pipeline de préprocessing complet
def preprocess_data(df):
    # 1. Supprimer duplicatas
    df = df.drop_duplicates()

    # 2. Gérer valeurs manquantes
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # 3. Encoder variables catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols)

    # 4. Normaliser features numériques
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
```

**Intégration avec ML:**
- Pandas pour préparation de données
- Conversion vers NumPy arrays pour entraînement
- Integration avec scikit-learn

```python
# Workflow typique
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Charger données
df = pd.read_csv('data.csv')

# 2. Préprocessing
df = preprocess_data(df)

# 3. Séparer features et target
X = df.drop('target', axis=1).values  # Conversion to NumPy
y = df['target'].values

# 4. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Entraîner modèle
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 3.3 Matplotlib & Seaborn (1 semaine)

**Concepts clés:**
- Visualisations de base (line, scatter, bar, histogram)
- Subplots et layouts
- Customisation (couleurs, labels, légendes)
- Visualisations statistiques avec Seaborn
- Visualisation de résultats ML

**Exercices pratiques:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Exercice 1: Visualisations basiques
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, y)
plt.title('Line Plot')

plt.subplot(1, 3, 2)
plt.scatter(x, y)
plt.title('Scatter Plot')

plt.subplot(1, 3, 3)
plt.hist(data, bins=30)
plt.title('Histogram')

plt.tight_layout()
plt.show()

# Exercice 2: Exploration de données
# Matrice de corrélation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Pairplot pour features
sns.pairplot(df, hue='target')

# Exercice 3: Visualisation de résultats ML
from sklearn.metrics import confusion_matrix

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')

# Courbe d'apprentissage
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
```

### Ressources Phase 3
- [NumPy Official Docs](https://numpy.org/doc/) - Documentation officielle
- [Pandas Documentation](https://pandas.pydata.org/docs/) - Guide complet
- [Medium - Data Preprocessing with NumPy and Pandas](https://medium.com/@arpitpathak114/data-preprocessing-with-numpy-and-pandas-5598ef69491e)
- [freeCodeCamp - Data Cleaning with Pandas](https://www.freecodecamp.org/news/data-cleaning-and-preprocessing-with-pandasbdvhj/)

### Projet intégré Phase 3
**Analyse exploratoire de données (EDA) complète:**
1. Charger dataset Titanic ou House Prices (Kaggle)
2. Nettoyer données (valeurs manquantes, outliers)
3. Feature engineering (créer nouvelles features)
4. Visualisations multiples (distributions, corrélations)
5. Statistiques descriptives par groupe
6. Préparer données pour modélisation

---

## 🤖 Phase 4: Machine Learning avec scikit-learn (6-8 semaines)

### Objectifs d'apprentissage
Comprendre et implémenter les algorithmes ML fondamentaux. Scikit-learn est l'outil essentiel pour ML traditionnel avant deep learning.

### 4.1 Fondamentaux ML (2 semaines)

**Concepts théoriques:**
- Apprentissage supervisé vs non-supervisé
- Problèmes de classification vs régression
- Train/validation/test splits
- Overfitting et underfitting
- Bias-variance tradeoff
- Cross-validation
- Métriques d'évaluation

**Workflow ML avec scikit-learn:**

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Préparation des données
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Entraînement
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train_scaled, y_train)

# 4. Évaluation
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 5. Cross-validation
scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 4.2 Algorithmes de Classification (2 semaines)

**Algorithmes à maîtriser:**
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

**Exercices pratiques:**

```python
# Exercice 1: Comparaison d'algorithmes
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    results[name] = score
    print(f"{name}: {score:.3f}")

# Exercice 2: Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Exercice 3: Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10],
         feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features')
```

### 4.3 Algorithmes de Régression (1 semaine)

**Algorithmes:**
- Linear Regression
- Ridge & Lasso Regression
- Polynomial Regression
- Random Forest Regressor
- Gradient Boosting

**Exercices:**

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# Exercice 1: Régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")

# Exercice 2: Régularisation (Ridge/Lasso)
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    print(f"Alpha {alpha}: R² = {score:.3f}")
```

### 4.4 Clustering (1 semaine)

**Algorithmes:**
- K-Means
- Hierarchical Clustering
- DBSCAN

**Exercices:**

```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Exercice 1: K-Means avec élbow method
inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X, kmeans.labels_))

plt.plot(K_range, inertias, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Exercice 2: Visualisation clusters
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0],
           kmeans.cluster_centers_[:, 1],
           s=300, c='red', marker='X')
plt.title('K-Means Clustering')
```

### 4.5 Pipelines ML (1 semaine)

**Concepts:**
- Pipeline sklearn
- Feature engineering dans pipelines
- Model persistence

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Exercice 1: Pipeline complet
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)

# Exercice 2: Pipeline avec feature engineering
from sklearn.preprocessing import PolynomialFeatures

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# Exercice 3: Sauvegarder et charger modèle
import joblib

# Sauvegarder
joblib.dump(pipeline, 'model.pkl')

# Charger
loaded_model = joblib.load('model.pkl')
predictions = loaded_model.predict(X_new)
```

### Projet intégré Phase 4
**Projet Kaggle - Prédiction de survie Titanic:**
1. EDA complète avec visualisations
2. Feature engineering (family size, titles)
3. Tester multiple algorithmes
4. Hyperparameter tuning
5. Ensemble methods (voting, stacking)
6. Submission Kaggle

### Ressources Phase 4
- [Scikit-learn Documentation](https://scikit-learn.org/) - Documentation officielle
- [Machine Learning Mastery](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/) - Guide step-by-step
- [DataCamp - Machine Learning Projects](https://www.datacamp.com/blog/machine-learning-projects-for-all-levels) - 33 projets pratiques
- [GeeksforGeeks ML Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/) - 100+ projets

---

## 🧠 Phase 5: Deep Learning avec PyTorch (8-12 semaines)

### Objectifs d'apprentissage
Maîtriser PyTorch pour implémenter des réseaux de neurones modernes. PyTorch est devenu le standard pour la recherche et le développement LLM.

### 5.1 Fondamentaux PyTorch (2 semaines)

**Concepts clés:**
- Tenseurs PyTorch vs NumPy arrays
- Autograd et backpropagation
- GPU acceleration (CUDA)
- Neural network building blocks

**Exercices fondamentaux:**

```python
import torch
import torch.nn as nn

# Exercice 1: Tenseurs basiques
x = torch.tensor([1., 2., 3.])
y = torch.tensor([4., 5., 6.])
z = x + y

# GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)

# Exercice 2: Autograd
x = torch.tensor([2.], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(f"dy/dx at x=2: {x.grad}")  # Devrait être 2*2 + 3 = 7

# Exercice 3: Réseau simple from scratch
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNet(10, 50, 2)
print(model)

# Exercice 4: Training loop basique
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 5.2 Computer Vision (3 semaines)

**Architectures:**
- Convolutional Neural Networks (CNN)
- ResNet, VGG, EfficientNet
- Transfer learning
- Data augmentation

**Projet 1: Classification MNIST**

```python
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Dataset et transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True,
                              transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Architecture CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters())

# Training loop complet
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%')
```

**Projet 2: Transfer Learning avec ResNet**

```python
import torchvision.models as models
from torchvision import transforms

# Charger modèle pré-entraîné
model = models.resnet50(pretrained=True)

# Geler couches pré-entraînées
for param in model.parameters():
    param.requires_grad = False

# Remplacer dernière couche
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Data augmentation
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Fine-tuning
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

### 5.3 NLP & Transformers (3 semaines)

**Concepts:**
- Embeddings
- Recurrent Neural Networks (RNN, LSTM, GRU)
- Attention mechanism
- Transformer architecture
- Tokenization

**Projet 3: Sentiment Analysis avec LSTM**

```python
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                           bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# Training
model = SentimentLSTM(vocab_size=10000, embedding_dim=100,
                     hidden_dim=256, output_dim=2)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
```

**Projet 4: Transformer basique**

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# Exercice: Implémenter encodeur transformer complet
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim,
                 num_layers, max_seq_length, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        seq_length = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_length, :]
        x = self.dropout(x)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)
```

### 5.4 Techniques avancées (2 semaines)

**Concepts:**
- Learning rate scheduling
- Gradient clipping
- Batch normalization & Layer normalization
- Early stopping
- Model checkpointing
- Mixed precision training
- Distributed training

```python
# Exercice 1: Learning rate scheduling
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

scheduler = CosineAnnealingLR(optimizer, T_max=100)
# ou
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

for epoch in range(num_epochs):
    train(model, train_loader)
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)

# Exercice 2: Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Exercice 3: Early stopping
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Exercice 4: Model checkpointing
def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

# Exercice 5: Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for epoch in range(num_epochs):
    for data, target in train_loader:
        optimizer.zero_grad()

        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
```

### Projet majeur Phase 5: FoodVision

**Description:** Construire un classificateur d'images de nourriture en 3 itérations progressives (milestone project du cours Learn PyTorch).

**Milestone 1: Classification basique**
- Dataset: Food-101 (101 classes de nourriture)
- Architecture: CNN custom simple
- Objectif: >60% accuracy

**Milestone 2: Transfer learning**
- Architecture: ResNet50 pré-entraîné
- Fine-tuning avec data augmentation
- Objectif: >80% accuracy

**Milestone 3: Production-ready**
- Optimisation hyperparamètres
- Model ensembling
- Déploiement avec API
- Objectif: >85% accuracy

### Ressources Phase 5
- [Learn PyTorch](https://www.learnpytorch.io/) - Cours Zero to Mastery complet
- [PyTorch Official Tutorials](https://docs.pytorch.org/tutorials/) - Documentation officielle
- [DataCamp - How to Learn PyTorch](https://www.datacamp.com/blog/how-to-learn-pytorch) - Guide 8 semaines
- [Medium - PyTorch Projects](https://medium.com/@vikasrahar007/getting-your-hands-dirty-pytorch-projects-from-basic-to-advanced-ba52ee9806a0) - Progression projets

---

## 🚀 Phase 6: Développement LLM (8-12 semaines)

### Objectifs d'apprentissage
Maîtriser les techniques spécifiques aux LLM: architecture, fine-tuning, RAG, déploiement. Devenir capable de construire et déployer des applications LLM.

### 6.1 Fondamentaux LLM (2 semaines)

**Concepts théoriques:**
- Architecture Transformer en profondeur
- Tokenization (BPE, WordPiece, SentencePiece)
- Positional encodings
- Attention mechanism détaillée
- Decoder-only vs Encoder-Decoder
- Scaling laws
- Emergent abilities

**Exercices théoriques:**

```python
# Exercice 1: Implémenter tokenizer BPE simple
class SimpleBPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {}

    def train(self, texts):
        # Compter paires de caractères
        # Merger les plus fréquentes itérativement
        pass

    def encode(self, text):
        # Tokenize text
        pass

    def decode(self, tokens):
        # Reconstruire text
        pass

# Exercice 2: Attention from scratch
import torch.nn.functional as F

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Args:
        query: (batch_size, num_heads, seq_len, d_k)
        key: (batch_size, num_heads, seq_len, d_k)
        value: (batch_size, num_heads, seq_len, d_v)
        mask: (batch_size, 1, seq_len, seq_len) ou None
    """
    d_k = query.size(-1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask if provided (pour causal attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attention_weights, value)

    return output, attention_weights

# Exercice 3: Multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))

        return output

# Exercice 4: GPT-style decoder block
class GPTBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention avec causal mask
        attn_output = self.attention(x, x, x, mask)
        x = self.ln1(x + self.dropout(attn_output))

        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.ln2(x + self.dropout(ff_output))

        return x
```

### 6.2 Hugging Face Transformers (2 semaines)

**Bibliothèque essentielle:** Hugging Face simplifie l'utilisation de modèles pré-entraînés.

**Exercices pratiques:**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Exercice 1: Charger et utiliser modèle pré-entraîné
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Génération de texte
prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs["input_ids"],
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exercice 2: Fine-tuning pour classification
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# Exercice 3: LoRA fine-tuning (Parameter-efficient)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # ~1% des paramètres!

# Exercice 4: Inference avec quantization
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 6.3 Prompt Engineering (1 semaine)

**Techniques:**
- Zero-shot prompting
- Few-shot prompting
- Chain-of-thought prompting
- ReAct (Reasoning + Acting)
- Prompt templates

```python
# Exercice 1: Few-shot learning
few_shot_prompt = """
Classify the sentiment of these movie reviews:

Review: "This movie was amazing! Best film of the year."
Sentiment: Positive

Review: "Terrible waste of time. Don't watch it."
Sentiment: Negative

Review: "It was okay, nothing special."
Sentiment: Neutral

Review: "{user_review}"
Sentiment:
"""

# Exercice 2: Chain-of-thought
cot_prompt = """
Question: {question}

Let's solve this step by step:
1. First, let's identify what we know
2. Then, let's break down the problem
3. Next, we'll solve each part
4. Finally, we'll combine the results

Answer:
"""

# Exercice 3: ReAct pattern
react_prompt = """
You have access to the following tools:
- search(query): Search the web
- calculate(expression): Evaluate math expressions
- python(code): Execute Python code

Question: {question}

Think step-by-step:
Thought 1: What information do I need?
Action 1: [choose tool and input]
Observation 1: [result from tool]

Thought 2: What's the next step?
Action 2: [choose tool and input]
Observation 2: [result from tool]

Final Answer: [your conclusion]
"""

# Exercice 4: Prompt template avec LangChain
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

template = """
Context: {context}

Question: {question}

Instructions: Provide a detailed answer based only on the context above.

Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(context=doc_text, question=user_query)
```

### 6.4 RAG (Retrieval-Augmented Generation) (2 semaines)

**Architecture RAG:** Combiner recherche sémantique + génération LLM.

**Pipeline complet:**

```python
# Exercice 1: Embeddings et vector store
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Créer embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = ["doc1 text...", "doc2 text...", "doc3 text..."]
embeddings = model.encode(documents)

# Index FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# Recherche
query = "What is machine learning?"
query_embedding = model.encode([query])
k = 3  # Top 3 résultats
distances, indices = index.search(query_embedding.astype('float32'), k)
relevant_docs = [documents[i] for i in indices[0]]

# Exercice 2: RAG avec LangChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# 1. Charger documents
loader = TextLoader("documents.txt")
documents = loader.load()

# 2. Splitter en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# 3. Créer embeddings et vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. Créer retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 5. Créer QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain({"query": "What is the main topic?"})
print(result["result"])
print("Sources:", result["source_documents"])

# Exercice 3: RAG avancé avec re-ranking
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, documents, top_k=3):
    # Score chaque document
    pairs = [[query, doc] for doc in documents]
    scores = cross_encoder.predict(pairs)

    # Trier par score
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    return [documents[i] for i in ranked_indices]

# Pipeline RAG complet
def rag_pipeline(query):
    # 1. Retrieval initial (large recall)
    initial_docs = vectorstore.similarity_search(query, k=10)

    # 2. Re-ranking (precision)
    reranked_docs = rerank_documents(query, initial_docs, top_k=3)

    # 3. Génération
    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    response = llm(prompt)

    return response, reranked_docs
```

### 6.5 Fine-tuning LLM (2 semaines)

**Techniques:**
- Full fine-tuning
- LoRA (Low-Rank Adaptation)
- QLoRA (Quantized LoRA)
- Instruction tuning
- RLHF basics

```python
# Exercice 1: Dataset preparation pour instruction tuning
instruction_dataset = [
    {
        "instruction": "Summarize the following text:",
        "input": "Long article text...",
        "output": "Summary of the article..."
    },
    # ...more examples
]

def format_instruction(sample):
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

# Exercice 2: Fine-tuning avec Trainer API
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

dataset = load_dataset("json", data_files="train.json")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    warmup_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)

trainer.train()

# Exercice 3: QLoRA fine-tuning (efficient!)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Charger modèle quantifié
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training (seulement ~0.2% des paramètres!)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# Sauvegarder adapters LoRA uniquement
model.save_pretrained("./lora_adapters")
```

### 6.6 Déploiement & Production (1-2 semaines)

**Outils:**
- FastAPI pour API
- Gradio pour demos
- Docker pour containerization
- Model optimization (quantization, pruning)

```python
# Exercice 1: API FastAPI simple
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

class GenerateResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=request.max_length,
        temperature=request.temperature
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return GenerateResponse(generated_text=text)

# Exercice 2: Gradio interface
import gradio as gr

def generate_text(prompt, max_length, temperature):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt"),
        gr.Slider(10, 200, value=100, label="Max Length"),
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="LLM Text Generator",
    description="Generate text using fine-tuned LLM"
)

interface.launch()

# Exercice 3: Model optimization
from optimum.onnxruntime import ORTModelForCausalLM

# Convertir en ONNX pour inference rapide
ort_model = ORTModelForCausalLM.from_pretrained(
    "model_path",
    export=True
)

# Quantization dynamique
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# Exercice 4: Dockerfile
"""
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# Exercice 5: Caching et batching
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_generate(prompt, max_length):
    # Cache responses pour prompts identiques
    return generate(prompt, max_length)

# Batching pour efficiency
def batch_generate(prompts, batch_size=8):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        outputs = model.generate(**inputs)
        texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        results.extend(texts)
    return results
```

### Projet majeur Phase 6: Chatbot personnalisé avec RAG

**Description:** Construire un chatbot conversationnel avec connaissance de domaine spécifique.

**Composantes:**
1. **Data collection**: Scraper documentation technique
2. **Preprocessing**: Chunking, cleaning
3. **Vector store**: Embeddings + FAISS/Chroma
4. **LLM**: Fine-tuner modèle base (LoRA)
5. **RAG pipeline**: Retrieval + generation
6. **Interface**: Gradio web UI
7. **Déploiement**: Docker + FastAPI

**Specs techniques:**
- Base model: Llama-2-7b ou Mistral-7b
- Fine-tuning: QLoRA sur conversations domaine
- Vector DB: ChromaDB persisté
- Reranking: Cross-encoder pour précision
- Memory: Historique conversation (sliding window)
- Evaluation: Rouge, BLEU, human eval

### Ressources Phase 6
- [GitHub - mlabonne/llm-course](https://github.com/mlabonne/llm-course) - Cours complet LLM avec roadmaps
- [Coursera - Generative AI with LLMs](https://www.coursera.org/learn/generative-ai-with-llms) - AWS & DeepLearning.AI
- [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) - Formation complète
- [MachineLearningMastery - LLM Roadmap 2025](https://machinelearningmastery.com/the-roadmap-for-mastering-language-models-in-2025/)

---

## 🛠️ Phase 7: Outils & Bonnes Pratiques (Transversal)

### 7.1 Environnements virtuels

**Pourquoi crucial:** Isoler dépendances par projet, éviter conflits de versions.

```bash
# venv (built-in Python)
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
ml_env\Scripts\activate  # Windows

pip install -r requirements.txt

# Conda (recommandé pour ML)
conda create -n ml_env python=3.10
conda activate ml_env
conda install pytorch torchvision -c pytorch
pip install transformers datasets

# Poetry (moderne)
poetry init
poetry add torch transformers
poetry install
```

**Best practice:** Un environnement par projet, fichier requirements.txt versionné.

### 7.2 Jupyter Notebooks

**Setup:**

```bash
# Installer Jupyter
pip install jupyter notebook jupyterlab

# Ajouter kernel depuis venv
pip install ipykernel
python -m ipykernel install --user --name=ml_env --display-name "Python (ML)"

# Lancer
jupyter lab
```

**Organisation notebook:**
```python
# 1. Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 2. Configuration
%matplotlib inline
%load_ext autoreload
%autoreload 2

pd.set_option('display.max_columns', None)

# 3. Chargement données
df = pd.read_csv('data.csv')

# 4. EDA
df.head()
df.info()

# 5. Visualisations
plt.figure(figsize=(10, 6))
# ...

# 6. Preprocessing
# ...

# 7. Modeling
# ...

# 8. Evaluation
# ...
```

**Bonnes pratiques:**
- Markdown cells pour documentation
- Restart kernel régulièrement
- Clear outputs avant commit git
- Exporter code vers .py pour production

### 7.3 Git pour ML

**Setup:**

```bash
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

**Gitignore pour ML:**

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/

# Jupyter
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# Data (ne jamais commit données sensibles!)
data/
*.csv
*.h5
*.pkl

# Models (trop gros pour git)
models/
*.pth
*.pt
*.onnx
checkpoints/

# Logs
logs/
*.log
tensorboard/

# IDE
.vscode/
.idea/
*.swp
```

**Workflow git:**

```bash
# Feature branch
git checkout -b feature/new-model

# Commits réguliers
git add .
git commit -m "Add data preprocessing pipeline"
git commit -m "Implement custom loss function"
git commit -m "Train model with new architecture"

# Push
git push origin feature/new-model

# Merge after PR review
git checkout main
git merge feature/new-model
```

**DVC (Data Version Control):** Pour versionner données et modèles.

```bash
pip install dvc

dvc init
dvc add data/large_dataset.csv
dvc add models/trained_model.pth

git add data/.gitignore data/large_dataset.csv.dvc
git commit -m "Track data with DVC"

# Push data to remote storage (S3, Google Drive, etc.)
dvc remote add -d storage s3://mybucket/dvcstore
dvc push
```

### 7.4 Testing

**Tests unitaires:**

```python
import unittest
import numpy as np

class TestPreprocessing(unittest.TestCase):
    def test_normalize(self):
        data = np.array([1, 2, 3, 4, 5])
        normalized = normalize(data)

        self.assertAlmostEqual(normalized.mean(), 0, places=5)
        self.assertAlmostEqual(normalized.std(), 1, places=5)

    def test_train_test_split(self):
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)

        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)

if __name__ == '__main__':
    unittest.main()
```

**Pytest (recommandé):**

```python
import pytest
import torch

def test_model_output_shape():
    model = MyModel(input_size=10, output_size=2)
    x = torch.randn(32, 10)
    output = model(x)
    assert output.shape == (32, 2)

def test_model_training():
    model = MyModel()
    loss_before = compute_loss(model, train_loader)

    train_one_epoch(model, train_loader)

    loss_after = compute_loss(model, train_loader)
    assert loss_after < loss_before

@pytest.fixture
def sample_data():
    return torch.randn(100, 10), torch.randint(0, 2, (100,))

def test_accuracy(sample_data):
    X, y = sample_data
    predictions = model.predict(X)
    accuracy = (predictions == y).float().mean()
    assert accuracy > 0.5
```

### 7.5 Documentation

**Docstrings:**

```python
def preprocess_text(text: str, lowercase: bool = True,
                   remove_punctuation: bool = True) -> str:
    """
    Preprocess text for NLP tasks.

    Args:
        text (str): Input text to preprocess
        lowercase (bool): Convert to lowercase if True (default: True)
        remove_punctuation (bool): Remove punctuation if True (default: True)

    Returns:
        str: Preprocessed text

    Examples:
        >>> preprocess_text("Hello, World!")
        'hello world'
        >>> preprocess_text("Hello, World!", lowercase=False)
        'Hello World'

    Raises:
        ValueError: If text is empty
    """
    if not text:
        raise ValueError("Text cannot be empty")

    if lowercase:
        text = text.lower()

    if remove_punctuation:
        text = ''.join(c for c in text if c.isalnum() or c.isspace())

    return text.strip()
```

**README.md structure:**

```markdown
# Project Title

Brief description of the project.

## Installation

\`\`\`bash
git clone https://github.com/user/project.git
cd project
pip install -r requirements.txt
\`\`\`

## Usage

\`\`\`python
from model import MyModel

model = MyModel()
predictions = model.predict(data)
\`\`\`

## Project Structure

\`\`\`
project/
├── data/               # Data directory
├── notebooks/          # Jupyter notebooks
├── src/
│   ├── models/        # Model definitions
│   ├── preprocessing/ # Data preprocessing
│   └── utils/         # Utility functions
├── tests/             # Unit tests
├── requirements.txt   # Dependencies
└── README.md
\`\`\`

## Training

\`\`\`bash
python train.py --config config.yaml
\`\`\`

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Baseline | 0.75 | 0.73 |
| Custom CNN | 0.89 | 0.88 |

## License

MIT
```

### 7.6 MLflow pour tracking experiments

```python
import mlflow
import mlflow.pytorch

# Setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("my-experiment")

# Training avec logging
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("epochs", 10)

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("config.yaml")

# Visualiser avec UI
# mlflow ui
```

### Ressources Phase 7
- [Using Virtual Environments in Jupyter](https://janakiev.com/blog/jupyter-virtual-envs/)
- [Best Practices for Jupyter Environments](https://www.zainrizvi.io/blog/jupyter-notebooks-best-practices-use-virtual-environments/)
- [Git for Data Science](https://www.datacamp.com/tutorial/git-push-pull)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

---

## 📚 Ressources complémentaires

### Cours en ligne
- **Andrew Ng - Machine Learning Specialization** (Coursera)
- **Fast.ai - Practical Deep Learning for Coders**
- **deeplearning.ai - Deep Learning Specialization**
- **Hugging Face Course** - NLP avec Transformers

### Livres recommandés
- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** - Aurélien Géron
- **"Deep Learning with PyTorch"** - Eli Stevens, Luca Antiga
- **"Natural Language Processing with Transformers"** - Lewis Tunstall et al.
- **"Building LLMs for Production"** - Sebastian Raschka

### Plateformes de pratique
- **Kaggle** - Compétitions et datasets
- **HackerEarth** - Challenges ML
- **Google Colab** - GPU gratuit pour prototypage
- **Hugging Face Spaces** - Déploiement demos

### Communautés
- **r/MachineLearning** (Reddit)
- **Hugging Face Forums**
- **PyTorch Discuss**
- **Fast.ai Forums**

---

## 🎯 Progression et évaluation

### Checkpoints d'auto-évaluation

**✅ Fin Phase 1 (Python Fondamental):**
- [ ] Écrire fonctions avec paramètres complexes
- [ ] Manipuler dictionnaires et listes imbriqués
- [ ] Gérer erreurs avec try/except
- [ ] Comprendre scope des variables

**✅ Fin Phase 2 (Python Intermédiaire):**
- [ ] Écrire list comprehensions complexes
- [ ] Créer decorators personnalisés
- [ ] Utiliser generators pour efficiency
- [ ] Implémenter context managers

**✅ Fin Phase 3 (Data Science):**
- [ ] Pipeline preprocessing NumPy complet
- [ ] Agrégations groupées Pandas
- [ ] Visualisations publication-ready
- [ ] EDA autonome sur nouveau dataset

**✅ Fin Phase 4 (ML avec scikit-learn):**
- [ ] Implémenter workflow ML complet
- [ ] Comparer multiple algorithmes
- [ ] Hyperparameter tuning efficace
- [ ] Créer pipelines réutilisables

**✅ Fin Phase 5 (Deep Learning PyTorch):**
- [ ] Entraîner CNN sur images
- [ ] Implémenter transformer from scratch
- [ ] Transfer learning productif
- [ ] Optimisation hyperparamètres avancée

**✅ Fin Phase 6 (LLM):**
- [ ] Fine-tuner LLM avec LoRA
- [ ] Implémenter RAG pipeline
- [ ] Prompt engineering efficace
- [ ] Déployer modèle en production

### Timeline réaliste

**Temps partiel (10h/semaine):**
- Phase 1-2: 6-8 semaines
- Phase 3: 6 semaines
- Phase 4: 8 semaines
- Phase 5: 12 semaines
- Phase 6: 12 semaines
- **Total: ~10-12 mois**

**Temps plein (40h/semaine):**
- Phase 1-2: 2 semaines
- Phase 3: 1.5 semaines
- Phase 4: 2 semaines
- Phase 5: 3 semaines
- Phase 6: 3 semaines
- **Total: ~3 mois**

### Conseils de progression

1. **Pratiquer quotidiennement** - Même 30min/jour vaut mieux que 3h sporadiques
2. **Projets > Théorie** - 70% pratique, 30% théorie
3. **Reproduire puis innover** - D'abord copier tutoriels, puis modifier, puis créer
4. **Kaggle competitions** - Benchmark réel de vos compétences
5. **Blog/GitHub** - Documenter apprentissage, construire portfolio
6. **Communauté** - Rejoindre forums, participer discussions
7. **Ne pas procrastiner sur "bases parfaites"** - Learn by doing

---

## 🎓 Certification et objectifs carrière

### Compétences après ce parcours

**Junior ML Engineer (6 mois):**
- Preprocessing données
- Entraîner modèles scikit-learn
- Fine-tuner modèles PyTorch
- Basique déploiement

**ML Engineer (12 mois):**
- Architecture complète ML pipelines
- Optimisation hyperparamètres avancée
- Production deployment
- Fine-tuning LLM

**LLM Developer (12 mois):**
- RAG pipelines sophistiqués
- Custom fine-tuning LoRA/QLoRA
- Prompt engineering avancé
- Production LLM applications

### Certifications utiles
- **TensorFlow Developer Certificate**
- **AWS Machine Learning Specialty**
- **Google Professional ML Engineer**
- **DeepLearning.AI Certifications**

---

## 📝 Notes finales

**Confidence levels:**
- Parcours structure: Haute confiance (basé sur consensus industrie)
- Durées estimées: Moyenne confiance (varient selon background)
- Exercices spécifiques: Haute confiance (testés communauté)
- Outils recommandés: Haute confiance (standards industrie 2024-2025)

**Limitations:**
- Exercices sont exemples, adaptez à votre niveau
- Timeline varie selon temps investi et background
- Technologies évoluent rapidement, vérifier versions
- LLM development est domaine récent, best practices émergentes

**Mise à jour:** 2025-11-27 - Basé sur standards actuels ML/LLM

---

## Sources

1. [GitHub - mlabonne/llm-course](https://github.com/mlabonne/llm-course)
2. [Data Science Roadmap - Scaler](https://www.scaler.com/blog/data-science-roadmap/)
3. [Zero to Mastery - Learn PyTorch](https://www.learnpytorch.io/)
4. [DataCamp - How to Learn PyTorch](https://www.datacamp.com/blog/how-to-learn-pytorch)
5. [Medium - Data Preprocessing with NumPy and Pandas](https://medium.com/@arpitpathak114/data-preprocessing-with-numpy-and-pandas-5598ef69491e)
6. [Real Python - Decorators Primer](https://realpython.com/primer-on-python-decorators/)
7. [Scientific Python Lectures](https://lectures.scientific-python.org/advanced/advanced_python/index.html)
8. [Using Virtual Environments in Jupyter](https://janakiev.com/blog/jupyter-virtual-envs/)
9. [Machine Learning Mastery - First ML Project](https://machinelearningmastery.com/machine-learning-in-python-step-by-step/)
10. [DataCamp - ML Projects for All Levels](https://www.datacamp.com/blog/machine-learning-projects-for-all-levels)
11. [GeeksforGeeks - ML Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/)
12. [Scikit-learn Official Documentation](https://scikit-learn.org/)
13. [MachineLearningMastery - LLM Roadmap 2025](https://machinelearningmastery.com/the-roadmap-for-mastering-language-models-in-2025/)
14. [Coursera - Generative AI with LLMs](https://www.coursera.org/learn/generative-ai-with-llms)
15. [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/)
