# LLM et Machine Learning en Finance Quantitative : Plan d'Apprentissage Complet

**Date de recherche**: 27 novembre 2025
**Niveau de confiance**: 85% (basé sur sources académiques et industrielles récentes)

---

## Table des Matières

1. [Vue d'ensemble exécutive](#vue-densemble-exécutive)
2. [Prérequis essentiels](#prérequis-essentiels)
3. [Concepts fondamentaux de finance quantitative](#concepts-fondamentaux-de-finance-quantitative)
4. [Applications ML en finance](#applications-ml-en-finance)
5. [Applications LLM spécifiques](#applications-llm-spécifiques)
6. [Datasets et APIs financières](#datasets-et-apis-financières)
7. [Frameworks spécialisés](#frameworks-spécialisés)
8. [Plan d'apprentissage structuré](#plan-dapprentissage-structuré)
9. [Projets pratiques](#projets-pratiques)
10. [Ressources et références](#ressources-et-références)

---

## Vue d'ensemble exécutive

La convergence du Machine Learning (ML) et des Large Language Models (LLM) avec la finance quantitative représente une révolution dans l'analyse financière et le trading algorithmique. Les recherches de 2024-2025 montrent des avancées significatives, notamment:

- **Performance LLM**: Les modèles GPT-3 (OPT) atteignent 74.4% de précision dans l'analyse de sentiment financier, surpassant FinBERT (72.2%) et les méthodes traditionnelles (50.1%)
- **Stratégies de trading**: Les stratégies long-short basées sur OPT génèrent un ratio de Sharpe de 3.05 et 355% de gains (août 2021 - juillet 2023)
- **Deep Learning**: Les architectures LSTM, CNN et modèles hybrides révolutionnent la prédiction de séries temporelles
- **RL Trading**: Les agents Deep Q-Learning (DQN) surpassent les modèles de base de 5% à 52% en rendements cumulés

---

## Prérequis essentiels

### 1. Mathématiques (Fondamental)

#### Niveau Basique
- **Statistiques descriptives**: moyenne, variance, corrélation
- **Distributions de probabilité**: Normale, Poisson, Exponentielle
- **Algèbre linéaire**: matrices, vecteurs, opérations matricielles
- **Calcul**: dérivées, intégrales, optimisation

#### Niveau Avancé
- **Calcul stochastique**: processus de Wiener, lemme d'Itô
- **Équations différentielles**: EDO et EDP pour la modélisation financière
- **Théorie des probabilités**: espérance conditionnelle, martingales
- **Analyse numérique**: méthodes de Monte Carlo, différences finies

**Ressources recommandées**:
- [MIT Mathematical Methods for Quantitative Finance](https://www.edx.org/learn/finance/massachusetts-institute-of-technology-mathematical-methods-for-quantitative-finance) (edX)
- Livres: "Options, Futures, and Other Derivatives" (Hull)

### 2. Statistiques et Économétrie (Fondamental)

#### Compétences requises
- **Régression**: linéaire, logistique, polynomiale
- **Analyse de séries temporelles**: ARIMA, GARCH, modèles de cointegration
- **Tests d'hypothèses**: p-values, intervalles de confiance
- **Techniques d'inférence**: maximum de vraisemblance, estimateurs bayésiens

**Importance**: La régression et l'analyse de séries temporelles sont la colonne vertébrale du trading quantitatif moderne.

### 3. Programmation (Essentiel)

#### Python (Priorité 1)
```python
# Bibliothèques essentielles
- numpy, pandas: manipulation de données
- matplotlib, seaborn: visualisation
- scikit-learn: ML classique
- tensorflow/pytorch: deep learning
- ta-lib: indicateurs techniques
- yfinance, pandas-datareader: données financières
```

#### Autres langages
- **C++/C#**: Pour le trading haute fréquence (50% du temps en implémentation)
- **R**: Analyse statistique et backtesting
- **SQL**: Gestion de bases de données financières

### 4. Connaissances financières (Important)

- Marchés financiers: actions, obligations, dérivés, forex
- Théorie du portefeuille: diversification, CAPM, frontière efficiente
- Gestion des risques: VaR, CVaR, stress testing
- Microstructure des marchés: carnet d'ordres, liquidité

---

## Concepts fondamentaux de finance quantitative

### 1. Analyse de séries temporelles

#### Modèles classiques
- **ARIMA** (AutoRegressive Integrated Moving Average): capture les patterns linéaires
- **GARCH** (Generalized AutoRegressive Conditional Heteroskedasticity): modélisation de la volatilité
- **GJR-GARCH**: asymétrie de la volatilité (marchés baissiers vs haussiers)

#### Applications Deep Learning
- **LSTM** (Long Short-Term Memory): capture les dépendances temporelles complexes
- **CNN** pour séries temporelles: extraction de features locaux
- **Transformers**: attention multi-tête pour séries temporelles longues
- **Modèles hybrides**: LSTM-ARIMA combinant forces complémentaires

**Recherche récente**: Les [Transformer Networks](https://blogs.mathworks.com/finance/2024/02/02/deep-learning-in-quantitative-finance-transformer-networks-for-time-series-prediction/) démontrent des performances supérieures pour la prédiction de prix, capturant des relations non-linéaires complexes.

### 2. Trading algorithmique

#### Stratégies principales

**1. High-Frequency Trading (HFT)**
- Exploitation de micro-mouvements de prix
- Systèmes ultra-rapides (<1ms latence)
- Colocation serveurs près des exchanges

**2. Statistical Arbitrage**
- Exploitation des mispricing statistiques
- Pair trading, mean reversion
- Cointegration entre actifs corrélés

**3. Mean Reversion**
- Prémisse: retour aux moyennes historiques
- Bandes de Bollinger, RSI, z-scores
- Risque: changements de régime de marché

**4. Momentum Trading**
- Continuation des tendances établies
- Moving averages, MACD, ADX
- Combinaison avec filtres de volatilité

#### Techniques avancées 2024-2025

Selon les [recherches récentes](https://www.sciencedirect.com/science/article/pii/S2590005625000177), les systèmes modernes intègrent:
- Deep Reinforcement Learning pour décisions adaptatives
- Multi-agent systems pour intelligence collective
- Hybrid architectures (CNN-LSTM-Attention)
- Real-time sentiment integration

---

## Applications ML en finance

### 1. Prédiction de prix

#### Modèles supervisés classiques
- **Random Forest**: robuste au surapprentissage, capture interactions non-linéaires
- **Gradient Boosting** (XGBoost, LightGBM): haute performance, interprétabilité via SHAP
- **SVM**: efficace en haute dimension, kernel tricks

#### Deep Learning avancé
- **LSTM Networks**:
  - Architecture: 50-200 unités, dropout 0.2-0.5
  - Sequence length: 20-60 timesteps
  - Applications: prédiction multi-horizon

- **CNN-LSTM Hybrides**:
  - CNN: extraction de features locaux (patterns courts)
  - LSTM: dépendances temporelles longues
  - Performance: +15-20% vs LSTM seul

- **Attention Mechanisms**:
  - Self-attention: pondération automatique de l'importance temporelle
  - Multi-head attention: capture de multiples patterns
  - Transformers: état de l'art pour séries longues

**Résultats empiriques**: Les modèles hybrides surpassent les méthodes traditionnelles de 20-40% en prédiction à court terme (1-5 jours).

### 2. Analyse de sentiment

#### Sources de données
- **News financières**: Bloomberg, Reuters, Financial Times
- **Réseaux sociaux**: Twitter (influenceurs), Reddit (r/wallstreetbets)
- **Rapports d'entreprise**: earnings calls, MD&A, 10-K/10-Q
- **Analyses d'analystes**: recommandations, target prices

#### Techniques NLP

**Approches traditionnelles**
- **Loughran-McDonald Dictionary**: lexique spécialisé finance (50.1% accuracy)
- **TF-IDF + classifiers**: baseline simple mais efficace

**Approches LLM modernes**
- **FinBERT**: BERT fine-tuné sur corpus financier
- **GPT-based models** (OPT): 74.4% accuracy sur sentiment
- **Ensemble methods**: combinaison de multiples modèles

**Performance comparative** ([source](https://www.sciencedirect.com/science/article/pii/S1544612324002575)):
```
OPT (GPT-3):      74.4% accuracy, Sharpe 3.05
BERT:             72.5% accuracy
FinBERT:          72.2% accuracy
L-M Dictionary:   50.1% accuracy
```

#### Pipeline d'analyse de sentiment
```python
1. Data Collection: scraping news/social media
2. Preprocessing: cleaning, tokenization, normalization
3. Sentiment Extraction: LLM inference or dictionary
4. Aggregation: time-weighted, volume-weighted
5. Signal Generation: sentiment scores → trading signals
6. Backtesting: historical performance evaluation
```

### 3. Détection de fraude et anomalies

#### Approches ML

**Supervisées** (59% des études)
- **SVM**: efficace pour patterns complexes
- **Neural Networks**: capture relations non-linéaires
- **Ensemble methods**: Random Forest, XGBoost pour réduire faux positifs

**Non-supervisées** (19% des études)
- **Isolation Forest**: détection d'outliers efficace
- **Autoencoders**: reconstruction errors pour anomalies
- **K-means clustering**: segmentation de comportements normaux/anormaux
- **One-Class SVM**: apprentissage de la normalité

**Semi-supervisées**
- Combinaison de peu de données labellisées + grandes quantités non-labellisées
- Efficace quand labels rares (fraud cases < 1%)

#### Techniques avancées

**Graph Neural Networks (GNN)**
- Modélisation de réseaux de transactions
- Détection de patterns de fraude en réseau
- Capacité: billions de records pour fraudes complexes

**Statistical Methods**
- Z-scores, déviations standard pour outliers
- Modèles probabilistes: HMM, Bayesian networks

**Challenges** ([source](https://www.nature.com/articles/s41599-024-03606-0)):
- Classification imprécise, coûts de misclassification disproportionnés
- Privacy concerns, performance computationnelle
- Évolution constante des patterns de fraude

### 4. Gestion de portefeuille

#### Modern Portfolio Theory (MPT)
- Optimisation moyenne-variance de Markowitz
- Frontière efficiente, ratio de Sharpe
- Limitations: hypothèses de normalité, corrélations statiques

#### ML-Enhanced Portfolio Management
- **Reinforcement Learning**: agents DQN pour allocation dynamique
- **Deep Learning**: prédiction de covariance matrices
- **Ensemble methods**: combinaison de multiples stratégies

---

## Applications LLM spécifiques

### 1. Analyse de news financières

#### Capacités des LLM

**Extraction d'information**
- Événements clés: M&A, earnings, guidance changes
- Entités: entreprises, produits, personnes, lieux
- Relations: causalités, corrélations, sentiments

**Génération de résumés**
- LongT5: résumés de documents longs (10-K, prospectus)
- Abstractive summarization: synthèse intelligente vs extractive

**Q&A financier**
- RAG (Retrieval-Augmented Generation): combinaison retrieval + generation
- FinGPT-RAG: framework spécialisé avec knowledge retrieval

#### Exemple de pipeline
```
1. News Ingestion: RSS feeds, APIs (Bloomberg, Reuters)
2. LLM Processing:
   - Sentiment: positive/negative/neutral + confidence
   - Entity extraction: tickers, sectors, events
   - Relationship extraction: cause-effect chains
3. Knowledge Graph: construction de graphe d'entités
4. Signal Generation: sentiment aggregation → trading signals
5. Risk Assessment: contradiction detection, uncertainty quantification
```

### 2. Génération de rapports financiers

#### Use cases

**Earnings Reports Analysis**
- Comparaison vs consensus, vs guidance
- Identification de surprises positives/négatives
- Trend analysis: YoY, QoQ comparisons

**Investment Research Reports**
- Génération automatique de rapports d'analyse
- Peer comparison, valuation multiples
- Incorporation de données alternatives (satellite, web traffic)

**Risk Reports**
- Stress testing scenarios, VaR calculations
- Regulatory reporting (Basel III, Solvency II)
- ESG assessment and reporting

### 3. Chatbots et assistants financiers

#### Applications client

**Personal Finance Assistants**
- Portfolio analysis, asset allocation advice
- Tax optimization suggestions
- Retirement planning, goal tracking

**Trading Assistants**
- Natural language trade execution: "Buy $1000 of AAPL"
- Market insights: "Why is TSLA down today?"
- Alert generation: price targets, technical signals

**Research Assistants**
- Company research: "Summarize MSFT's last 3 earnings calls"
- Competitive analysis: "Compare GOOGL vs MSFT cloud revenue growth"
- Macro analysis: "How does Fed policy affect tech stocks?"

#### Architectures
- **Retrieval-Augmented Generation (RAG)**: combine retrieval + LLM generation
- **Fine-tuned models**: adaptation à domaines spécifiques
- **Multi-agent systems**: spécialisation par expertise

---

## Datasets et APIs financières

### 1. Yahoo Finance

#### Caractéristiques
- **Gratuité**: accès libre aux données historiques
- **Historique**: jusqu'à 10 ans de données
- **Couverture**: actions mondiales, indices, forex, crypto, commodities
- **Fréquence**: daily, weekly, monthly (pas d'intraday gratuit)
- **Rate limits**: ~2000 requests/heure par IP (~48k/jour)

#### Accès Python
```python
import yfinance as yf

# Download single stock
data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')

# Multiple tickers
data = yf.download(['AAPL', 'GOOGL', 'MSFT'], period='1y')

# Ticker object for more info
ticker = yf.Ticker('AAPL')
info = ticker.info  # company info
financials = ticker.financials  # financial statements
```

#### Limitations
- API non officielle (peut changer sans préavis)
- Qualité de données variable pour marchés non-US
- Pas de données haute fréquence

### 2. Alpha Vantage

#### Caractéristiques
- **API officielle**: support et documentation
- **Free tier**: 500 requests/jour (25 requests/jour pour premium data)
- **Données**: 20+ ans d'historique, real-time intraday
- **Indicateurs techniques**: 50+ indicateurs intégrés (SMA, EMA, RSI, MACD, Bollinger)
- **Données alternatives**: crypto, forex, commodities
- **Sentiment**: news sentiment powered by AI/ML

#### Accès Python
```python
from alpha_vantage.timeseries import TimeSeries

# Initialize with API key
ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')

# Daily data
data, meta_data = ts.get_daily('AAPL', outputsize='full')

# Intraday data
data, meta = ts.get_intraday('MSFT', interval='5min', outputsize='full')

# Technical indicators
from alpha_vantage.techindicators import TechIndicators
ti = TechIndicators(key='YOUR_API_KEY', output_format='pandas')
data, meta = ti.get_sma('AAPL', interval='daily', time_period=20)
```

#### Pricing
- Free: 500 req/jour
- Premium ($49.99/mois): 1200 req/minute, extended history
- Documentation: [Alpha Vantage API Docs](https://www.alphavantage.co/documentation/)

### 3. Quandl (Nasdaq Data Link)

#### Caractéristiques
- **Datasets diversifiés**: millions de datasets financiers, économiques, alternatifs
- **Qualité**: données curées, provenant d'exchanges officiels, gouvernements
- **Coverage**: stocks, options, futures, forex, commodities, économie, données alternatives
- **Historique**: décennies de données pour certaines séries
- **API unifiée**: accès standardisé à multiples sources

#### Accès Python
```python
import quandl

# Set API key
quandl.ApiConfig.api_key = 'YOUR_API_KEY'

# Download data
data = quandl.get('WIKI/AAPL', start_date='2020-01-01', end_date='2024-12-31')

# Multiple datasets
data = quandl.get(['WIKI/AAPL.4', 'WIKI/GOOGL.4'])  # closing prices

# Economic data
gdp = quandl.get('FRED/GDP')  # US GDP from Federal Reserve
```

#### Datasets populaires
- WIKI Prices: US stock prices (historique, discontinued)
- Sharadar: US equities fundamental data
- Futures: CME, ICE futures contracts
- Economics: FRED, World Bank, UN data

### 4. Autres APIs importantes

#### IEX Cloud
- **Free tier**: 50,000 messages/mois
- **Données**: real-time US equities, historical, news
- **Spécialité**: market microstructure, tick data
- Site: [IEX Cloud](https://iexcloud.io/)

#### Twelve Data
- **Caractéristiques**: real-time + historical, 100+ exchanges
- **Free tier**: 800 API calls/jour
- **Support**: REST API, WebSocket, Python SDK

#### Polygon.io
- **Focus**: granularité élevée, tick-by-tick data
- **Pricing**: à partir de $29/mois
- **Data**: stocks, options, forex, crypto

#### CCXT (Crypto)
- **Open-source**: bibliothèque Python unifiée
- **Exchanges**: 100+ crypto exchanges
- **Données**: OHLCV, order book, trades

```python
import ccxt

exchange = ccxt.binance()
ticker = exchange.fetch_ticker('BTC/USDT')
ohlcv = exchange.fetch_ohlcv('ETH/USDT', '1h', limit=100)
```

---

## Frameworks spécialisés

### 1. FinBERT

#### Caractéristiques
- **Base**: BERT pré-entraîné, fine-tuné sur corpus financier
- **Dataset**: Financial PhraseBank (Malo et al., 2014)
- **Outputs**: softmax pour 3 labels (positive, negative, neutral)
- **Performance**: 72.2% accuracy sur sentiment financier
- **Disponibilité**: [Hugging Face](https://huggingface.co/ProsusAI/finbert), open-source

#### Utilisation
```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model
tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

# Predict sentiment
text = "Stocks rallied and the British pound gained."
inputs = tokenizer(text, return_tensors="pt", padding=True)
outputs = model(**inputs)
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Labels: positive, negative, neutral
labels = ['positive', 'negative', 'neutral']
sentiment = labels[predictions.argmax()]
```

#### Avantages
- Spécialisé finance, meilleur que BERT générique
- Open-source, reproductible
- Intégration facile avec Hugging Face

### 2. BloombergGPT

#### Caractéristiques
- **Taille**: 50 milliards de paramètres
- **Dataset**: 363B tokens financiers (Bloomberg archives 40 ans) + 345B tokens généraux
- **Training**: ~1.3M GPU hours sur NVIDIA A100 (coût $1-2M)
- **Performance**: SOTA sur tâches financières spécialisées
- **Accès**: propriétaire Bloomberg, non public

#### Capacités
- Analyse de documents financiers complexes
- Extraction d'entités et relations financières
- Q&A sur données financières
- Génération de rapports et résumés

#### Limitations
- Non accessible hors Bloomberg
- Coût de training prohibitif pour reproduction
- Nécessité de re-training fréquent pour fraîcheur (très coûteux)

### 3. FinGPT

#### Caractéristiques
- **Philosophy**: open-source, démocratisation de l'IA financière
- **Avantage**: lightweight adaptation, fine-tuning rapide et peu coûteux
- **Coût**: <$300 par fine-tuning (vs $1-2M pour BloombergGPT)
- **Updates**: fine-tuning mensuel/hebdomadaire avec nouvelles données
- **Disponibilité**: [GitHub](https://github.com/AI4Finance-Foundation/FinGPT), modèles sur HuggingFace

#### Architecture
- Base: LLaMA, GPT-style models
- Fine-tuning: LoRA, QLoRA pour efficacité
- Datasets: financial news, social media, reports

#### Use cases
- Sentiment analysis temps réel
- Portfolio management agents
- Financial Q&A systems
- Market prediction

#### Utilisation
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "AI4Finance-Foundation/FinGPT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Inference
prompt = "Analyze the sentiment of the following news: ..."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
response = tokenizer.decode(outputs[0])
```

### 4. Autres frameworks notables

#### FinRL (Financial Reinforcement Learning)
- **Focus**: RL for trading
- **Algorithms**: DQN, A2C, PPO, SAC, TD3
- **Environments**: custom gym environments pour trading
- **GitHub**: [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)

#### QuantLib
- **Type**: bibliothèque C++ pour pricing de dérivés
- **Capacités**: options pricing, risk management, yield curves
- **Python binding**: QuantLib-Python

#### PyAlgoTrade
- **Type**: backtesting framework
- **Features**: event-driven, technical indicators
- **Support**: stocks, forex, bitcoin

#### Backtrader
- **Type**: backtesting et trading live
- **Features**: multiples data feeds, indicators, optimisation
- **Community**: large, bien documenté

---

## Plan d'apprentissage structuré

### Phase 1: Fondations (2-3 mois)

#### Semaines 1-4: Mathématiques et statistiques
**Objectifs**:
- Maîtriser statistiques descriptives et inférentielles
- Comprendre distributions de probabilité
- Algèbre linéaire de base (matrices, vecteurs)

**Ressources**:
- Khan Academy: Statistics and Probability
- 3Blue1Brown: Essence of Linear Algebra (YouTube)
- Cours: [Statistics for Data Science](https://www.coursera.org/learn/statistics-for-data-science-python) (Coursera)

**Exercices pratiques**:
```python
# Calculer statistiques sur données boursières
import yfinance as yf
import numpy as np

data = yf.download('AAPL', period='1y')
returns = data['Close'].pct_change().dropna()

print(f"Mean return: {returns.mean():.4f}")
print(f"Std dev: {returns.std():.4f}")
print(f"Sharpe ratio: {returns.mean() / returns.std() * np.sqrt(252):.2f}")

# Visualiser distribution
import matplotlib.pyplot as plt
returns.hist(bins=50)
plt.title('Distribution of AAPL daily returns')
plt.show()
```

#### Semaines 5-8: Python et manipulation de données
**Objectifs**:
- Maîtriser NumPy, Pandas
- Visualisation avec Matplotlib, Seaborn
- Nettoyage et préparation de données

**Ressources**:
- [Python for Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) (gratuit en ligne)
- Kaggle Learn: Python, Pandas, Data Visualization

**Projet**:
```python
# Analyser portefeuille de 5 actions
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
weights = [0.2, 0.2, 0.2, 0.2, 0.2]

# Download data
data = yf.download(tickers, period='2y')['Adj Close']

# Calculate portfolio returns
returns = data.pct_change()
portfolio_returns = (returns * weights).sum(axis=1)

# Performance metrics
cumulative_returns = (1 + portfolio_returns).cumprod()
sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()

print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Max Drawdown: {max_drawdown:.2%}")
```

#### Semaines 9-12: ML classique
**Objectifs**:
- Comprendre supervised vs unsupervised learning
- Maîtriser scikit-learn
- Régression, classification, clustering

**Ressources**:
- [Scikit-learn tutorials](https://scikit-learn.org/stable/tutorial/index.html)
- Cours: [Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning) (Coursera)

**Projet**: Prédiction de direction de prix (classification)
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Feature engineering
data = yf.download('AAPL', period='5y')
data['Return'] = data['Close'].pct_change()
data['Target'] = (data['Return'].shift(-1) > 0).astype(int)  # next day up/down

# Technical indicators
data['SMA_20'] = data['Close'].rolling(20).mean()
data['SMA_50'] = data['Close'].rolling(50).mean()
data['RSI'] = calculate_rsi(data['Close'], 14)  # implement or use ta-lib
data.dropna(inplace=True)

# Features
features = ['Return', 'SMA_20', 'SMA_50', 'RSI', 'Volume']
X = data[features]
y = data['Target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))
```

### Phase 2: Finance quantitative (2-3 mois)

#### Semaines 13-16: Séries temporelles financières
**Objectifs**:
- Maîtriser ARIMA, GARCH
- Stationnarité, cointegration
- Backtesting de stratégies

**Ressources**:
- [Quantitative Economics with Python](https://quantecon.org/lectures/) (QuantEcon)
- Livre: "Analysis of Financial Time Series" (Tsay)

**Projet**: Mean reversion strategy
```python
from statsmodels.tsa.stattools import coint
import backtrader as bt

# Find cointegrated pairs
def find_cointegrated_pairs(data):
    n = data.shape[1]
    pvalue_matrix = np.ones((n, n))

    for i in range(n):
        for j in range(i+1, n):
            result = coint(data.iloc[:, i], data.iloc[:, j])
            pvalue_matrix[i, j] = result[1]

    return pvalue_matrix

# Download data for multiple stocks
tickers = ['XLE', 'XLF', 'XLU', 'XLK', 'XLV']
data = yf.download(tickers, period='2y')['Adj Close']

# Find pairs
pvalue_matrix = find_cointegrated_pairs(data)
# Select most cointegrated pair (lowest p-value)

# Implement pair trading strategy with backtrader
class PairTradingStrategy(bt.Strategy):
    def __init__(self):
        self.zscore = None
        self.upper_threshold = 2.0
        self.lower_threshold = -2.0

    def next(self):
        # Calculate spread and z-score
        spread = self.datas[0] - self.hedge_ratio * self.datas[1]
        zscore = (spread - spread.mean()) / spread.std()

        # Trading logic
        if zscore > self.upper_threshold:
            # Short spread
            self.sell(data=self.datas[0])
            self.buy(data=self.datas[1])
        elif zscore < self.lower_threshold:
            # Long spread
            self.buy(data=self.datas[0])
            self.sell(data=self.datas[1])
```

#### Semaines 17-20: Algorithmic trading
**Objectifs**:
- Stratégies momentum, mean reversion
- Risk management, position sizing
- Backtesting avec Backtrader/Zipline

**Ressources**:
- [QuantStart](https://www.quantstart.com/articles/) (articles gratuits)
- GitHub: [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)

**Projet**: Multi-strategy portfolio
```python
import backtrader as bt

class MomentumStrategy(bt.Strategy):
    params = (('momentum_period', 20),)

    def __init__(self):
        self.momentum = bt.indicators.ROC(
            self.data.close,
            period=self.params.momentum_period
        )

    def next(self):
        if self.momentum[0] > 0:
            if not self.position:
                self.buy()
        elif self.momentum[0] < 0:
            if self.position:
                self.sell()

# Run backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MomentumStrategy)
data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2020,1,1))
cerebro.adddata(data)
cerebro.broker.setcash(100000.0)
cerebro.run()
print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
```

#### Semaines 21-24: Portfolio optimization
**Objectifs**:
- Modern Portfolio Theory
- Efficient frontier, Sharpe ratio optimization
- Risk parity, Black-Litterman

**Ressources**:
- [PyPortfolioOpt documentation](https://pyportfolioopt.readthedocs.io/)
- Cours: [Computational Finance and Financial Econometrics](https://www.coursera.org/learn/computational-finance-mathematical-models)

**Projet**: Optimisation de portefeuille
```python
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Download data
tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'JNJ', 'V', 'PG', 'NVDA']
data = yf.download(tickers, period='3y')['Adj Close']

# Calculate returns and covariance
mu = expected_returns.mean_historical_return(data)
S = risk_models.sample_cov(data)

# Optimize for maximum Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("Optimal weights:")
for ticker, weight in cleaned_weights.items():
    print(f"{ticker}: {weight:.2%}")

# Performance
ef.portfolio_performance(verbose=True)

# Discrete allocation
latest_prices = get_latest_prices(data)
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=10000)
allocation, leftover = da.lp_portfolio()
print(f"\nDiscrete allocation: {allocation}")
print(f"Funds remaining: ${leftover:.2f}")
```

### Phase 3: Deep Learning pour finance (2-3 mois)

#### Semaines 25-28: Deep Learning fondations
**Objectifs**:
- Neural networks basics
- TensorFlow/PyTorch
- CNN, RNN, LSTM architectures

**Ressources**:
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (Andrew Ng, Coursera)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)

**Projet**: LSTM pour prédiction de prix
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        X = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len]
        return torch.FloatTensor(X), torch.FloatTensor([y])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Prepare data
data = yf.download('AAPL', period='5y')['Close'].values
data = (data - data.mean()) / data.std()  # normalize

# Train/test split
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create datasets
train_dataset = StockDataset(train_data, seq_len=60)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch.unsqueeze(-1))
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
```

#### Semaines 29-32: NLP et LLM pour finance
**Objectifs**:
- Transformers, attention mechanism
- BERT, GPT architectures
- Fine-tuning pour finance

**Ressources**:
- [Hugging Face NLP Course](https://huggingface.co/course)
- [FinBERT paper](https://arxiv.org/abs/1908.10063)

**Projet**: Sentiment analysis avec FinBERT
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from datasets import Dataset

# Load FinBERT
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare custom dataset (financial news + labels)
# Format: {'text': "news text", 'label': 0/1/2}  # negative/neutral/positive
df = pd.read_csv('financial_news_dataset.csv')
dataset = Dataset.from_pandas(df)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Fine-tuning
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# Inference
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ['negative', 'neutral', 'positive']
    sentiment = labels[probs.argmax().item()]
    confidence = probs.max().item()
    return sentiment, confidence

# Test
news = "Apple reported record quarterly revenue driven by strong iPhone sales"
sentiment, conf = predict_sentiment(news)
print(f"Sentiment: {sentiment} (confidence: {conf:.2%})")
```

#### Semaines 33-36: Reinforcement Learning pour trading
**Objectifs**:
- RL basics: MDP, Q-learning, policy gradients
- DQN, PPO, A2C algorithms
- Trading environments

**Ressources**:
- [Spinning Up in Deep RL](https://spinningup.openai.com/) (OpenAI)
- [FinRL Library](https://github.com/AI4Finance-Foundation/FinRL)

**Projet**: DQN trading agent
```python
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.df = df
        self.initial_balance = initial_balance
        self.current_step = 0

        # Action space: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)

        # Observation space: [balance, shares_held, price, volume, indicators...]
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_value = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            self.balance / self.initial_balance,
            self.shares_held,
            row['Close'] / 100,  # normalize
            row['Volume'] / 1e9,
            row['SMA_20'],
            row['SMA_50'],
            row['RSI'] / 100,
            row['MACD'],
            row['Signal'],
            self.total_value / self.initial_balance
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['Close']

        # Execute action
        if action == 0:  # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price * 0.999  # 0.1% fee
                self.shares_held = 0
        elif action == 2:  # Buy
            shares_to_buy = self.balance // (current_price * 1.001)
            if shares_to_buy > 0:
                self.shares_held += shares_to_buy
                self.balance -= shares_to_buy * current_price * 1.001

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Calculate reward
        self.total_value = self.balance + self.shares_held * current_price
        reward = (self.total_value - self.initial_balance) / self.initial_balance

        return self._get_observation(), reward, done, {}

    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance:.2f}, '
              f'Shares: {self.shares_held}, Total Value: {self.total_value:.2f}')

# Prepare environment
data = yf.download('AAPL', period='3y')
# Add indicators (SMA, RSI, MACD, etc.)
# ... feature engineering code ...

env = TradingEnv(data)
env = DummyVecEnv([lambda: env])

# Train DQN agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=0.005,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1
)

model.learn(total_timesteps=100000)

# Test agent
obs = env.reset()
for i in range(len(data)):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, info = env.step(action)
    env.render()
    if done:
        break
```

### Phase 4: Projets avancés et spécialisation (2-3 mois)

#### Semaines 37-40: Système de trading complet
**Objectif**: Intégrer tous les composants dans un système production-ready

**Architecture système**:
```
┌─────────────────────────────────────────────────────────┐
│                    Trading System                       │
├─────────────────────────────────────────────────────────┤
│  Data Layer                                             │
│  ├─ Market Data (Alpha Vantage, Yahoo Finance)         │
│  ├─ Alternative Data (News, Social Media)              │
│  └─ Fundamental Data (Financial statements)            │
├─────────────────────────────────────────────────────────┤
│  Feature Engineering                                    │
│  ├─ Technical Indicators (TA-Lib)                      │
│  ├─ Sentiment Scores (FinBERT)                         │
│  └─ Fundamental Metrics                                │
├─────────────────────────────────────────────────────────┤
│  ML Models                                              │
│  ├─ Price Prediction (LSTM, Transformers)              │
│  ├─ Sentiment Analysis (FinBERT, GPT)                  │
│  ├─ Regime Detection (HMM, Clustering)                 │
│  └─ Portfolio Optimization (RL, CVaR)                  │
├─────────────────────────────────────────────────────────┤
│  Risk Management                                        │
│  ├─ Position Sizing (Kelly Criterion)                  │
│  ├─ Stop Loss / Take Profit                            │
│  └─ VaR / CVaR Monitoring                              │
├─────────────────────────────────────────────────────────┤
│  Execution                                              │
│  ├─ Order Management                                   │
│  ├─ Broker Integration (Alpaca, Interactive Brokers)   │
│  └─ Slippage / Transaction Costs                       │
├─────────────────────────────────────────────────────────┤
│  Monitoring & Logging                                   │
│  ├─ Performance Metrics                                │
│  ├─ Model Drift Detection                              │
│  └─ Alerts & Notifications                             │
└─────────────────────────────────────────────────────────┘
```

**Composants clés**:
```python
# config.py
class Config:
    # API Keys
    ALPHA_VANTAGE_KEY = 'your_key'
    TWITTER_API_KEY = 'your_key'

    # Trading parameters
    INITIAL_CAPITAL = 100000
    MAX_POSITION_SIZE = 0.1  # 10% per position
    STOP_LOSS_PCT = 0.02  # 2% stop loss

    # Model parameters
    LSTM_LOOKBACK = 60
    SENTIMENT_WINDOW = 7  # days
    REBALANCE_FREQ = 'weekly'

# data_manager.py
class DataManager:
    def __init__(self):
        self.alpha_vantage = AlphaVantage(Config.ALPHA_VANTAGE_KEY)
        self.news_api = NewsAPI(Config.NEWS_API_KEY)

    def get_market_data(self, tickers, start_date, end_date):
        """Fetch OHLCV data"""
        pass

    def get_news_sentiment(self, ticker, days=7):
        """Fetch and analyze news sentiment"""
        pass

    def get_fundamentals(self, ticker):
        """Fetch financial statements"""
        pass

# feature_engineer.py
class FeatureEngineer:
    def add_technical_indicators(self, df):
        """Add SMA, RSI, MACD, Bollinger Bands, etc."""
        pass

    def add_sentiment_features(self, df, ticker):
        """Add sentiment scores from news"""
        pass

    def add_fundamental_features(self, df, ticker):
        """Add PE ratio, debt/equity, etc."""
        pass

# model_manager.py
class ModelManager:
    def __init__(self):
        self.price_predictor = LSTMModel()
        self.sentiment_analyzer = FinBERT()
        self.regime_detector = HMM()

    def predict_price(self, features):
        """Predict next day price"""
        pass

    def analyze_sentiment(self, text):
        """Analyze news sentiment"""
        pass

    def detect_regime(self, market_data):
        """Detect market regime (bull/bear/sideways)"""
        pass

# risk_manager.py
class RiskManager:
    def calculate_position_size(self, signal_strength, volatility):
        """Kelly criterion or volatility-based sizing"""
        pass

    def check_risk_limits(self, portfolio):
        """Ensure within risk limits"""
        pass

    def calculate_var(self, portfolio, confidence=0.95):
        """Value at Risk calculation"""
        pass

# execution_engine.py
class ExecutionEngine:
    def __init__(self, broker_api):
        self.broker = broker_api

    def execute_trades(self, signals):
        """Execute buy/sell orders"""
        pass

    def rebalance_portfolio(self, target_weights):
        """Rebalance to target allocation"""
        pass

# main.py
class TradingSystem:
    def __init__(self):
        self.data_manager = DataManager()
        self.feature_engineer = FeatureEngineer()
        self.model_manager = ModelManager()
        self.risk_manager = RiskManager()
        self.execution_engine = ExecutionEngine()

    def run_strategy(self):
        while True:
            # 1. Fetch data
            data = self.data_manager.get_market_data(tickers, ...)

            # 2. Engineer features
            features = self.feature_engineer.add_all_features(data)

            # 3. Generate signals
            predictions = self.model_manager.predict_price(features)
            sentiment = self.model_manager.analyze_sentiment(...)
            regime = self.model_manager.detect_regime(data)

            # 4. Combine signals
            signals = self.combine_signals(predictions, sentiment, regime)

            # 5. Risk management
            sized_positions = self.risk_manager.calculate_positions(signals)

            # 6. Execute
            self.execution_engine.execute_trades(sized_positions)

            # 7. Monitor
            self.log_performance()

            # Sleep until next execution
            time.sleep(Config.EXECUTION_INTERVAL)
```

#### Semaines 41-44: Multi-agent trading system
**Projet**: Système avec agents spécialisés

```python
# Inspired by: https://www.sciencedirect.com/science/article/abs/pii/S0957417422013082

class TradingAgent:
    """Base class for trading agents"""
    def __init__(self, name, timeframe):
        self.name = name
        self.timeframe = timeframe
        self.rl_agent = DQN(...)  # or PPO, A2C

    def analyze(self, data):
        """Agent-specific analysis"""
        pass

    def generate_signal(self, state):
        """Generate buy/sell/hold signal"""
        action, _ = self.rl_agent.predict(state)
        return action

class ShortTermAgent(TradingAgent):
    """Focuses on 1-5 day horizon"""
    def __init__(self):
        super().__init__("ShortTerm", "1D")
        # Emphasize technical indicators

    def analyze(self, data):
        features = {
            'rsi': calculate_rsi(data),
            'macd': calculate_macd(data),
            'volume_spike': detect_volume_spike(data)
        }
        return features

class MediumTermAgent(TradingAgent):
    """Focuses on 1-4 week horizon"""
    def __init__(self):
        super().__init__("MediumTerm", "1W")
        # Balance technical and fundamental

    def analyze(self, data):
        features = {
            'trend': identify_trend(data),
            'support_resistance': find_levels(data),
            'earnings_momentum': calculate_earnings_surprise(data)
        }
        return features

class LongTermAgent(TradingAgent):
    """Focuses on 1-6 month horizon"""
    def __init__(self):
        super().__init__("LongTerm", "1M")
        # Emphasize fundamentals and sentiment

    def analyze(self, data):
        features = {
            'value_metrics': calculate_value_metrics(data),
            'growth_metrics': calculate_growth_metrics(data),
            'sentiment_trend': calculate_sentiment_trend(data)
        }
        return features

class MetaAgent:
    """Coordinates multiple agents and makes final decisions"""
    def __init__(self):
        self.agents = [
            ShortTermAgent(),
            MediumTermAgent(),
            LongTermAgent()
        ]
        self.ensemble_model = None  # DQN or meta-learner

    def aggregate_signals(self, data):
        """Collect signals from all agents"""
        signals = []
        for agent in self.agents:
            features = agent.analyze(data)
            signal = agent.generate_signal(features)
            signals.append({
                'agent': agent.name,
                'signal': signal,
                'confidence': agent.get_confidence()
            })
        return signals

    def make_decision(self, signals):
        """Meta-decision using ensemble or voting"""
        # Option 1: Weighted voting
        weights = [0.3, 0.5, 0.2]  # short, medium, long
        weighted_signal = sum(s['signal'] * w for s, w in zip(signals, weights))

        # Option 2: Meta-learner (another RL agent)
        meta_state = self.construct_meta_state(signals)
        final_action = self.ensemble_model.predict(meta_state)

        return final_action

    def train_agents(self, historical_data):
        """Train each agent on its specific timeframe"""
        for agent in self.agents:
            env = TradingEnv(historical_data, timeframe=agent.timeframe)
            agent.rl_agent.learn(total_timesteps=100000)

# Usage
meta_agent = MetaAgent()
meta_agent.train_agents(historical_data)

# Live trading
while True:
    current_data = fetch_market_data()
    signals = meta_agent.aggregate_signals(current_data)
    decision = meta_agent.make_decision(signals)
    execute_trade(decision)
```

#### Semaines 45-48: Advanced topics et recherche

**Topics avancés**:

1. **Attention-based models pour finance**
```python
# Transformer for time series
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # last timestep
        return out
```

2. **Graph Neural Networks pour asset relationships**
```python
import torch_geometric as pyg

class FinancialGNN(nn.Module):
    """Model asset relationships as graph"""
    def __init__(self, num_features, hidden_dim=64):
        super().__init__()
        self.conv1 = pyg.nn.GCNConv(num_features, hidden_dim)
        self.conv2 = pyg.nn.GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x

# Build graph: nodes=assets, edges=correlations
def build_asset_graph(returns_df, threshold=0.5):
    corr_matrix = returns_df.corr()
    edge_index = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
    return torch.tensor(edge_index).t()
```

3. **Explainable AI pour trading decisions**
```python
import shap

# SHAP for model interpretation
def explain_prediction(model, X):
    explainer = shap.TreeExplainer(model)  # for tree models
    shap_values = explainer.shap_values(X)

    # Visualize
    shap.summary_plot(shap_values, X)
    shap.force_plot(explainer.expected_value, shap_values[0], X.iloc[0])

    return shap_values

# Attention visualization for Transformers
def visualize_attention(model, input_sequence):
    attention_weights = model.get_attention_weights(input_sequence)
    plt.imshow(attention_weights, cmap='viridis')
    plt.xlabel('Input timesteps')
    plt.ylabel('Output timesteps')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.show()
```

4. **Advanced risk management**
```python
import cvxpy as cp

# CVaR optimization (risk-aware)
def optimize_portfolio_cvar(returns, alpha=0.95, target_return=0.10):
    n_assets = returns.shape[1]
    n_scenarios = returns.shape[0]

    # Variables
    weights = cp.Variable(n_assets)
    z = cp.Variable(n_scenarios)
    var = cp.Variable()

    # CVaR formulation
    portfolio_returns = returns @ weights
    cvar = var + (1/(1-alpha)) * cp.sum(z) / n_scenarios

    # Constraints
    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        z >= 0,
        z >= -(portfolio_returns - var),
        cp.mean(portfolio_returns) >= target_return
    ]

    # Minimize CVaR
    problem = cp.Problem(cp.Minimize(cvar), constraints)
    problem.solve()

    return weights.value, cvar.value
```

---

## Projets pratiques

### Projet 1: Bot de sentiment trading (Débutant)

**Objectif**: Trading bot simple basé sur analyse de sentiment des news

**Compétences**: Web scraping, NLP, backtesting

**Stack technologique**:
- Python: pandas, numpy, matplotlib
- NLP: FinBERT ou TextBlob
- Data: yfinance, NewsAPI
- Backtesting: Backtrader

**Étapes**:
1. **Data collection**: scraper news financières (NewsAPI, Finnhub)
2. **Sentiment analysis**: utiliser FinBERT pour scorer sentiment
3. **Signal generation**: convertir scores en signaux (buy/sell/hold)
4. **Backtesting**: tester sur données historiques
5. **Visualisation**: plot performance, equity curve

**Code de démarrage**:
```python
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline

# Initialize
newsapi = NewsApiClient(api_key='YOUR_KEY')
sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_news_sentiment(ticker, days=1):
    """Fetch news and calculate average sentiment"""
    news = newsapi.get_everything(
        q=ticker,
        language='en',
        sort_by='publishedAt',
        from_param=f'{days} days ago'
    )

    sentiments = []
    for article in news['articles']:
        text = article['title'] + ' ' + article['description']
        result = sentiment_pipeline(text[:512])[0]
        score = 1 if result['label'] == 'positive' else -1 if result['label'] == 'negative' else 0
        sentiments.append(score)

    return np.mean(sentiments) if sentiments else 0

def generate_signal(sentiment_score, threshold=0.3):
    """Convert sentiment to trading signal"""
    if sentiment_score > threshold:
        return 'BUY'
    elif sentiment_score < -threshold:
        return 'SELL'
    else:
        return 'HOLD'

# Main loop
ticker = 'AAPL'
for date in trading_dates:
    sentiment = get_news_sentiment(ticker)
    signal = generate_signal(sentiment)

    if signal == 'BUY':
        # Execute buy logic
        pass
    elif signal == 'SELL':
        # Execute sell logic
        pass
```

**Ressources**:
- Tutorial: [Sentiment Trading Bot](https://www.udemy.com/course/sentiment-trading-python/)
- GitHub: [stocks-trading-bot](https://github.com/GunjanDhanuka/stocks-trading-bot)

### Projet 2: LSTM Price Predictor (Intermédiaire)

**Objectif**: Prédiction de prix avec deep learning

**Compétences**: Deep learning, feature engineering, model evaluation

**Stack technologique**:
- PyTorch/TensorFlow
- Technical indicators: TA-Lib
- Hyperparameter tuning: Optuna
- MLOps: MLflow

**Étapes**:
1. **Feature engineering**: créer indicateurs techniques
2. **Data preparation**: windowing, normalization, train/val/test split
3. **Model architecture**: LSTM → Attention → Dense
4. **Training**: avec early stopping, learning rate scheduling
5. **Hyperparameter tuning**: optimiser architecture et hyperparamètres
6. **Evaluation**: MAE, RMSE, direction accuracy, backtesting
7. **Deployment**: sauvegarder modèle, créer API inference

**Architecture avancée**:
```python
class AdvancedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # Attention layer
        self.attention = nn.MultiheadAttention(
            hidden_size * 2,  # bidirectional
            num_heads=4
        )

        # Dense layers
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Take last timestep
        out = attn_out[:, -1, :]

        # Dense layers
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
```

**Hyperparameter tuning avec Optuna**:
```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    hidden_size = trial.suggest_int('hidden_size', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1e-2)

    # Train model
    model = AdvancedLSTM(input_size, hidden_size, num_layers, dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop...
    val_loss = train_and_evaluate(model, optimizer, train_loader, val_loader)

    return val_loss

# Optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best params: {study.best_params}")
```

### Projet 3: Multi-Strategy Portfolio (Avancé)

**Objectif**: Système de trading combinant multiples stratégies et ML

**Compétences**: System design, MLOps, production deployment

**Stack technologique**:
- Orchestration: Airflow
- Models: Ensemble (LSTM, XGBoost, RL agents)
- Monitoring: Prometheus, Grafana
- Deployment: Docker, Kubernetes
- Backtesting: Zipline, Backtrader

**Architecture**:
```
┌─────────────────────────────────────────────┐
│           Orchestrator (Airflow)            │
├─────────────────────────────────────────────┤
│  Strategy 1: Mean Reversion                 │
│  ├─ Model: ARIMA + Statistical tests        │
│  └─ Signals: Pairs trading, RSI extremes    │
├─────────────────────────────────────────────┤
│  Strategy 2: Momentum                       │
│  ├─ Model: LSTM + Trend detection           │
│  └─ Signals: Breakouts, moving avg cross    │
├─────────────────────────────────────────────┤
│  Strategy 3: Sentiment                      │
│  ├─ Model: FinBERT + Topic modeling         │
│  └─ Signals: News sentiment, social media   │
├─────────────────────────────────────────────┤
│  Strategy 4: RL Agent                       │
│  ├─ Model: Multi-agent DQN                  │
│  └─ Signals: Learned optimal policies       │
├─────────────────────────────────────────────┤
│  Meta-Strategy Allocator                    │
│  ├─ Ensemble: Voting + Stacking             │
│  ├─ Risk Management: CVaR optimization      │
│  └─ Position Sizing: Kelly criterion        │
├─────────────────────────────────────────────┤
│  Execution & Monitoring                     │
│  ├─ Broker: Alpaca API                      │
│  ├─ Monitoring: Real-time dashboards        │
│  └─ Alerts: Performance, risk breaches      │
└─────────────────────────────────────────────┘
```

**Airflow DAG pour orchestration**:
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'trading_system',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'trading_pipeline',
    default_args=default_args,
    schedule_interval='0 9 * * 1-5',  # Every weekday at 9 AM
    catchup=False
)

# Tasks
fetch_data = PythonOperator(
    task_id='fetch_market_data',
    python_callable=fetch_all_data,
    dag=dag
)

engineer_features = PythonOperator(
    task_id='engineer_features',
    python_callable=create_features,
    dag=dag
)

# Parallel strategy execution
mean_reversion = PythonOperator(
    task_id='mean_reversion_strategy',
    python_callable=run_mean_reversion,
    dag=dag
)

momentum = PythonOperator(
    task_id='momentum_strategy',
    python_callable=run_momentum,
    dag=dag
)

sentiment = PythonOperator(
    task_id='sentiment_strategy',
    python_callable=run_sentiment,
    dag=dag
)

rl_agent = PythonOperator(
    task_id='rl_agent_strategy',
    python_callable=run_rl_agent,
    dag=dag
)

# Meta-strategy
aggregate_signals = PythonOperator(
    task_id='aggregate_signals',
    python_callable=combine_strategies,
    dag=dag
)

optimize_portfolio = PythonOperator(
    task_id='optimize_portfolio',
    python_callable=run_optimization,
    dag=dag
)

execute_trades = PythonOperator(
    task_id='execute_trades',
    python_callable=place_orders,
    dag=dag
)

# Dependencies
fetch_data >> engineer_features >> [mean_reversion, momentum, sentiment, rl_agent]
[mean_reversion, momentum, sentiment, rl_agent] >> aggregate_signals
aggregate_signals >> optimize_portfolio >> execute_trades
```

**Monitoring dashboard avec Prometheus + Grafana**:
```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Metrics
trades_executed = Counter('trades_executed_total', 'Total trades executed')
portfolio_value = Gauge('portfolio_value_usd', 'Current portfolio value in USD')
strategy_performance = Gauge('strategy_performance', 'Strategy performance', ['strategy_name'])
prediction_latency = Histogram('prediction_latency_seconds', 'Model prediction latency')

# Update metrics
def execute_trade(order):
    trades_executed.inc()
    # ... trade logic ...

def update_portfolio_value(value):
    portfolio_value.set(value)

def record_strategy_perf(strategy_name, perf):
    strategy_performance.labels(strategy_name=strategy_name).set(perf)

# Start Prometheus server
start_http_server(8000)
```

### Projet 4: Financial News Analysis System (Expert)

**Objectif**: Système complet d'analyse de news avec LLM

**Composants**:
1. News scraping & aggregation
2. LLM-based analysis (FinGPT, GPT-4)
3. Knowledge graph construction
4. Trading signal generation
5. Real-time dashboard

**Technologies**:
- LLM: OpenAI API, FinGPT
- Graph DB: Neo4j
- Real-time: Kafka, WebSockets
- Frontend: React, D3.js

**Pipeline**:
```python
class NewsAnalysisSystem:
    def __init__(self):
        self.scraper = NewsScraper()
        self.llm = FinGPT()
        self.graph_db = Neo4jConnection()
        self.signal_generator = SignalGenerator()

    def process_news(self, article):
        # 1. Extract entities and events
        entities = self.llm.extract_entities(article['text'])
        events = self.llm.extract_events(article['text'])

        # 2. Sentiment analysis
        sentiment = self.llm.analyze_sentiment(article['text'])

        # 3. Impact assessment
        impact = self.llm.assess_market_impact(article['text'], entities)

        # 4. Store in knowledge graph
        self.graph_db.add_article(article, entities, events, sentiment, impact)

        # 5. Generate trading signals
        signals = self.signal_generator.generate_from_news(
            entities, events, sentiment, impact
        )

        return signals

    def run_continuous(self):
        """Continuous news monitoring"""
        while True:
            new_articles = self.scraper.fetch_latest()

            for article in new_articles:
                try:
                    signals = self.process_news(article)
                    self.publish_signals(signals)
                except Exception as e:
                    logger.error(f"Error processing article: {e}")

            time.sleep(60)  # Check every minute
```

---

## Ressources et références

### Livres recommandés

**Finance quantitative**:
1. "Advances in Financial Machine Learning" - Marcos López de Prado
2. "Machine Learning for Algorithmic Trading" - Stefan Jansen
3. "Quantitative Trading" - Ernest Chan
4. "Options, Futures, and Other Derivatives" - John Hull
5. "Python for Finance" - Yves Hilpisch

**Machine Learning**:
1. "Deep Learning" - Goodfellow, Bengio, Courville
2. "Hands-On Machine Learning" - Aurélien Géron
3. "Reinforcement Learning" - Sutton and Barto

### Cours en ligne

**MOOCs**:
- [Machine Learning for Trading](https://www.udacity.com/course/machine-learning-for-trading--ud501) - Georgia Tech (Udacity)
- [Computational Finance](https://www.coursera.org/specializations/computational-finance) - University of Washington (Coursera)
- [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Andrew Ng (Coursera)
- [Financial Engineering and Risk Management](https://www.coursera.org/specializations/financial-engineering-risk-management) - Columbia (Coursera)

**Plateformes spécialisées**:
- [QuantStart](https://www.quantstart.com/) - Tutorials et articles
- [QuantInsti](https://www.quantinsti.com/) - EPAT program
- [Quantopian Lectures](https://www.quantopian.com/lectures) (archivé mais utile)

### GitHub repositories essentiels

1. **[stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading)**
   - Code du livre "ML for Algorithmic Trading"
   - Exemples complets de stratégies

2. **[AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT)**
   - FinGPT models et exemples
   - Fine-tuning guides

3. **[AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL)**
   - RL for finance
   - Multiple algorithms (DQN, PPO, A2C, SAC)

4. **[cbailes/awesome-deep-trading](https://github.com/cbailes/awesome-deep-trading)**
   - Curated list of DL trading resources

### Papiers académiques clés

**LLM for Finance** (2024-2025):
- [Sentiment trading with large language models](https://www.sciencedirect.com/science/article/pii/S1544612324002575)
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)
- [Large Language Models in equity markets](https://pmc.ncbi.nlm.nih.gov/articles/PMC12421730/)

**Deep Learning for Trading**:
- [Deep Learning in Quantitative Trading](https://www.cambridge.org/core/elements/abs/deep-learning-in-quantitative-trading/C39DE06D255470F6232BC97E2E5474E7)
- [Deep learning for algorithmic trading: A systematic review](https://www.sciencedirect.com/science/article/pii/S2590005625000177)

**Reinforcement Learning**:
- [Multi-agent deep reinforcement learning for algorithmic trading](https://www.sciencedirect.com/science/article/abs/pii/S0957417422013082)
- [Deep Q-learning for stock market forecasting](https://www.sciencedirect.com/science/article/abs/pii/S0957417420306321)

### Communautés et forums

- **QuantConnect**: Forum et communauté de quants
- **r/algotrading**: Reddit community
- **r/MachineLearning**: ML discussions
- **Elite Trader**: Forum de trading
- **Wilmott Forums**: Quant finance discussions

### Datasets publics

**Financial data**:
- Quandl/Nasdaq Data Link
- Yahoo Finance (via yfinance)
- Alpha Vantage
- WRDS (Wharton, academic)

**Alternative data**:
- Reddit WallStreetBets data
- Twitter financial sentiment datasets
- SEC EDGAR filings
- News APIs (NewsAPI, Finnhub)

**Research datasets**:
- [Kaggle Financial datasets](https://www.kaggle.com/datasets?search=finance)
- [UCI ML Repository - Finance](https://archive.ics.uci.edu/ml/index.php)

---

## Conclusion et prochaines étapes

### Roadmap personnalisée selon profil

**Profil Débutant** (Background non-technique):
- **Durée**: 12-18 mois
- **Focus**: Fondations solides (math, stats, Python)
- **Premier projet**: Sentiment trading bot
- **Objectif**: Comprendre concepts fondamentaux et workflow complet

**Profil Intermédiaire** (Background CS/Math):
- **Durée**: 6-12 mois
- **Focus**: Deep learning, stratégies avancées
- **Premier projet**: LSTM price predictor
- **Objectif**: Maîtriser ML/DL pour finance, déployer système production

**Profil Avancé** (Background ML/Finance):
- **Durée**: 3-6 mois
- **Focus**: LLM, RL, systèmes multi-agents
- **Premier projet**: Multi-strategy portfolio
- **Objectif**: Recherche, publications, systèmes enterprise-grade

### Compétences clés à maîtriser (par priorité)

**🔴 Essentielles**:
1. Python (NumPy, Pandas, Matplotlib)
2. Statistiques et probabilités
3. Machine Learning classique (scikit-learn)
4. Backtesting frameworks
5. APIs de données financières

**🟡 Importantes**:
6. Deep Learning (PyTorch/TensorFlow)
7. Séries temporelles (ARIMA, GARCH)
8. NLP et transformers
9. Reinforcement Learning
10. Risk management

**🟢 Avancées**:
11. Attention mechanisms, Transformers
12. Graph Neural Networks
13. Multi-agent systems
14. MLOps et deployment
15. Explainable AI

### Pièges courants à éviter

1. **Overfitting**: Trop optimiser sur données historiques
   - Solution: Validation croisée robuste, walk-forward analysis

2. **Look-ahead bias**: Utiliser future information dans backtests
   - Solution: Stricte séparation temporelle des données

3. **Survivorship bias**: Ignorer actifs délistés
   - Solution: Utiliser datasets complets incluant survivorship

4. **Transaction costs**: Ignorer frais et slippage
   - Solution: Modéliser coûts réalistes (0.1-0.5% par trade)

5. **Data snooping**: Tester trop de stratégies sur mêmes données
   - Solution: Out-of-sample testing, paper trading

### Métriques de succès

**Phase d'apprentissage**:
- Complétion de projets pratiques ✓
- Understanding conceptuel solide ✓
- Code reproductible et propre ✓

**Phase de développement**:
- Backtests positifs sur >5 ans ✓
- Sharpe ratio >1.5 ✓
- Max drawdown <20% ✓

**Phase de production**:
- Paper trading rentable sur 3-6 mois ✓
- Système robuste et automatisé ✓
- Monitoring et alertes opérationnels ✓

### Ressources pour aller plus loin

**Conférences**:
- NeurIPS (Machine Learning)
- ICML (Machine Learning)
- AAAI (AI)
- QWAFi (Quantitative Finance)

**Certifications**:
- CQF (Certificate in Quantitative Finance)
- FRM (Financial Risk Manager)
- CFA (Chartered Financial Analyst)

**Recherche avancée**:
- arXiv.org - section q-fin (Quantitative Finance)
- SSRN - Financial Research Network
- Google Scholar - finance + machine learning queries

---

## Annexe: Code templates et snippets

### Template: Backtesting framework
```python
class Backtester:
    def __init__(self, strategy, data, initial_capital=100000):
        self.strategy = strategy
        self.data = data
        self.initial_capital = initial_capital
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.holdings = {}
        self.equity_curve = []
        self.trades = []

    def run(self):
        for i in range(len(self.data)):
            # Get current state
            current_data = self.data.iloc[:i+1]

            # Generate signal
            signal = self.strategy.generate_signal(current_data)

            # Execute trades
            if signal['action'] == 'BUY':
                self.execute_buy(signal)
            elif signal['action'] == 'SELL':
                self.execute_sell(signal)

            # Record portfolio value
            portfolio_value = self.calculate_portfolio_value()
            self.equity_curve.append(portfolio_value)

        return self.calculate_metrics()

    def execute_buy(self, signal):
        price = signal['price']
        shares = int(self.cash * signal['size'] / price)
        cost = shares * price * 1.001  # 0.1% transaction cost

        if cost <= self.cash:
            self.cash -= cost
            self.holdings[signal['ticker']] = self.holdings.get(signal['ticker'], 0) + shares
            self.trades.append({
                'date': signal['date'],
                'action': 'BUY',
                'ticker': signal['ticker'],
                'shares': shares,
                'price': price
            })

    def execute_sell(self, signal):
        ticker = signal['ticker']
        if ticker in self.holdings and self.holdings[ticker] > 0:
            shares = signal.get('shares', self.holdings[ticker])
            proceeds = shares * signal['price'] * 0.999  # 0.1% transaction cost

            self.cash += proceeds
            self.holdings[ticker] -= shares
            self.trades.append({
                'date': signal['date'],
                'action': 'SELL',
                'ticker': ticker,
                'shares': shares,
                'price': signal['price']
            })

    def calculate_portfolio_value(self):
        total = self.cash
        for ticker, shares in self.holdings.items():
            current_price = self.get_current_price(ticker)
            total += shares * current_price
        return total

    def calculate_metrics(self):
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1

        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

        cummax = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'final_value': equity_curve[-1]
        }
```

### Template: Model training pipeline
```python
class ModelTrainingPipeline:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.best_model = None
        self.best_loss = float('inf')

    def prepare_data(self, data):
        # Feature engineering
        features = self.engineer_features(data)

        # Train/val/test split (temporal)
        train_size = int(len(features) * 0.7)
        val_size = int(len(features) * 0.15)

        train_data = features[:train_size]
        val_data = features[train_size:train_size+val_size]
        test_data = features[train_size+val_size:]

        return train_data, val_data, test_data

    def train(self, train_data, val_data):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5
        )

        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss = self.train_epoch(train_data, optimizer)

            # Validation
            val_loss = self.evaluate(val_data)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Early stopping
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model = copy.deepcopy(self.model)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

            # Logging
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        return self.best_model

    def train_epoch(self, data, optimizer):
        self.model.train()
        total_loss = 0

        for batch in data:
            optimizer.zero_grad()
            predictions = self.model(batch['features'])
            loss = self.criterion(predictions, batch['targets'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data)

    def evaluate(self, data):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in data:
                predictions = self.model(batch['features'])
                loss = self.criterion(predictions, batch['targets'])
                total_loss += loss.item()

        return total_loss / len(data)
```

---

**Fin du rapport de recherche**

*Ce document a été généré à partir de recherches approfondies menées en novembre 2025. Les informations sont basées sur des sources académiques et industrielles récentes. Pour les mises à jour, consulter les sources citées.*

**Niveau de confiance global**: 85%
**Dernière mise à jour**: 27 novembre 2025
