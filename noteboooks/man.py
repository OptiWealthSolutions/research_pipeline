import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import sys



TICKER = "EURUSD=X"                
PERIOD = "2y"                     
INTERVAL = "1h"                   
TARGET_SHIFT_PERIODS = 1       
TEST_SIZE_RATIO = 0.3         




def load_data(ticker, period, interval):
    """
    Charge les données OHLCV depuis yfinance et les nettoie.
    """
    print(f"Chargement des données pour {ticker} (Période: {period}, Interval: {interval})...")
    df = yf.download(tickers=ticker, period=period, interval=interval)
    
    if df.empty:
        print(f"ERREUR: Aucune donnée retournée par yfinance pour {ticker}.")
        sys.exit(1)
        

    df.columns = df.columns.str.lower()
    
    required_cols = ['open', 'high', 'low', 'Close']
    if not all(col in df.columns for col in required_cols):
        print(f"ERREUR: Données manquantes. Nécessite {required_cols}.")
        sys.exit(1)
        
    return df



def calculer_ma_feature(data):
    """
    Calcule la feature à tester.
    PREND: un DataFrame pandas (data) avec les colonnes OHLC.
    RETOURNE: (pandas.Series, str) -> (série de la feature, nom de la feature)
    """
    
    period = 14
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    perte = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / perte
    rsi = 100 - (100 / (1 + rs))
    
    feature_series = rsi
    feature_name = "RSI_14"


    if not isinstance(feature_series, pd.Series):
        raise TypeError("La feature doit être une pandas.Series.")
        
    return feature_series.rename(feature_name), feature_name


def define_target(df, periods):
    """
    Crée la target binaire (1 = Hausse, 0 = Baisse/Stagnation).
    """
    df['future_Close'] = df['Close'].shift(-periods)
    df['target'] = (df['future_Close'] > df['Close']).astype(int)
    return df

def prepare_data(df, feature_name):
    """
    Sélectionne la feature et la target, et nettoie les NaN.
    """
    df_analysis = df[[feature_name, 'target']].copy()
    
    initial_rows = len(df_analysis)
    df_analysis = df_analysis.dropna()
    final_rows = len(df_analysis)
    
    print(f"Préparation des données: {initial_rows - final_rows} lignes supprimées (NaNs).")
    print(f"Taille finale du jeu de données: {final_rows} lignes.")
    
    return df_analysis



def plot_correlation(df_analysis, feature_name):
    """
    Affiche l'heatmap de corrélation.
    """
    correlation = df_analysis.corr()
    print("\n--- Matrice de Corrélation ---")
    print(correlation)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(correlation, annot=True, cmap='vlag', vmin=-1, vmax=1)
    plt.title(f"Corrélation entre {feature_name} et la Target")
    plt.show()

def plot_distribution(df_analysis, feature_name):
    """
    Affiche le boxplot de la distribution de la feature vs la target.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='target', y=feature_name, data=df_analysis, palette='vlag')
    plt.title(f"Distribution de {feature_name} pour Target=0 vs Target=1")
    plt.xlabel("Target (0 = Baisse, 1 = Hausse)")
    plt.ylabel(f"Valeur de {feature_name}")
    plt.show()

def plot_quantile_analysis(df_analysis, feature_name):
    """
    Affiche l'analyse du taux de succès par quantile.
    """
    try:
        df_analysis['feature_quantile'] = pd.qcut(df_analysis[feature_name], 5, labels=False, duplicates='drop')
        quantile_analysis = df_analysis.groupby('feature_quantile')['target'].mean()

        print("\n--- Taux de succès (moyenne de Target) par Quantile ---")
        print(quantile_analysis)

        quantile_analysis.plot(kind='bar', figsize=(10, 5))
        plt.title(f"Taux de succès (Prob. de Hausse) par Quantile de {feature_name}")
        plt.ylabel("Moyenne de la Target (Taux de succès)")
        plt.xlabel(f"Quantile de {feature_name} (0 = Plus Bas, 4 = Plus Haut)")
        plt.axhline(df_analysis['target'].mean(), color='r', linestyle='--', label='Taux de succès global')
        plt.legend()
        plt.show()

    except ValueError as e:
        print(f"AVERTISSEMENT (Quantiles): {e}. (Arrive si la feature a trop peu de valeurs uniques.)")


def run_model_test(df_analysis, feature_name, test_size):
    """
    Entraîne et évalue un modèle de Régression Logistique simple.
    """
    X = df_analysis[[feature_name]]
    y = df_analysis['target']

    if len(X) < 50:
        print("ERREUR: Pas assez de données pour la modélisation (après nettoyage).")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    print(f"\nTaille Train: {len(X_train)} | Taille Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    benchmark_accuracy = y_test.value_counts(normalize=True).max()

    print("\n--- ÉVALUATION SUR LE SET DE TEST ---")
    print(f"Précision (Accuracy) du modèle: {accuracy:.4f}")
    print(f"Précision de base (Benchmark):   {benchmark_accuracy:.4f}")

    if accuracy > benchmark_accuracy:
        print("\n>> VERDICT: Le modèle bat le benchmark. La feature semble avoir un pouvoir prédictif.")
    else:
        print("\n>> VERDICT: Le modèle NE BAT PAS le benchmark. La feature semble peu ou pas explicative.")

    print("\n--- Rapport de Classification ---")
    print(classification_report(y_test, y_pred))

    print("\n--- Matrice de Confusion ---")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Prédit Baisse (0)', 'Prédit Hausse (1)'],
                yticklabels=['Actuel Baisse (0)', 'Actuel Hausse (1)'])
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.show()


def main():
    """
    Orchestre l'exécution du script.
    """
    df = load_data(TICKER, PERIOD, INTERVAL)
    
    df_feature, feature_name = calculer_ma_feature(df)
    df[feature_name] = df_feature
    print(f"Feature '{feature_name}' calculée.")
    
    df = define_target(df, TARGET_SHIFT_PERIODS)
    
    df_analysis = prepare_data(df, feature_name)
    
    if df_analysis.empty:
        print("ERREUR: Le DataFrame est vide après nettoyage. Arrêt.")
        sys.exit(1)
        
    plot_correlation(df_analysis, feature_name)
    plot_distribution(df_analysis, feature_name)
    plot_quantile_analysis(df_analysis, feature_name)
    
    run_model_test(df_analysis, feature_name, TEST_SIZE_RATIO)
    
    print("\n--- Analyse terminée. ---")

if __name__ == "__main__":
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    main()