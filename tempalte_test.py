import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_features(data):
    """
    C'EST ICI QUE VOUS AJOUTEZ VOTRE LOGIQUE DE FEATURE.
    En exemple, nous créons une feature aléatoire.
    Remplacez np.random.randn par votre propre calcul (ex: RSI, Spread...).
    """
    data['my_feature'] = np.random.randn(len(data))
    # data['votre_autre_feature'] = ...
    return data

def analyze_feature_impact(ticker='EURUSD=X', start_date='2015-01-01', end_date='2024-12-31', target_days=3):
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            print(f"Aucune donnée téléchargée pour {ticker}.")
            return
    except Exception as e:
        print(f"Erreur de téléchargement : {e}")
        return

    data = create_features(data)
    
    target_col = f'target_return_{target_days}d'
    # Création de la colonne cible
    data[target_col] = data['Close'].pct_change(target_days).shift(-target_days)
    
    # Nettoyage des NaN
    data = data.dropna()
    
    if data.empty:
        print("Le DataFrame est vide après la création des features et de la cible. Vérifiez vos calculs et périodes.")
        return
    
    # VÉRIFICATION CRITIQUE : Assurer que la colonne cible est présente après dropna()
    if target_col not in data.columns:
        print(f"ERREUR CRITIQUE: La colonne cible '{target_col}' est manquante dans le DataFrame après dropna().")
        print(f"Colonnes disponibles: {data.columns.tolist()}")
        return
        
    feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', target_col]]
    
    if not feature_columns:
        print("Aucune feature n'a été trouvée. Modifiez la fonction create_features.")
        return

    for feature in feature_columns:
        if not pd.api.types.is_numeric_dtype(data[feature]):
            continue
            
        print(f"--- Analyse pour la Feature : {feature} ---")
        
        # 1. Analyse de Corrélation
        plt.figure(figsize=(14, 6))
        
        # Utilisation de target_col pour la clarté
        corr_matrix = data[[feature, target_col]].corr()
        correlation = corr_matrix.loc[feature, target_col]
        
        ax1 = plt.subplot(1, 2, 1)
        sns.heatmap(corr_matrix, annot=True, cmap='vlag', center=0, ax=ax1)
        ax1.set_title(f'Matrice de Corrélation (Corrélation: {correlation:.3f})')
        
        # 2. Scatter Plot
        ax2 = plt.subplot(1, 2, 2)
        sns.regplot(data=data, x=feature, y=target_col, scatter_kws={'alpha':0.2, 's':10}, line_kws={'color':'red'}, ax=ax2)
        ax2.set_title(f'Relation entre {feature} et {target_col}')
        
        plt.tight_layout()
        plt.show()

        # 3. Analyse par Quantiles
        try:
            data[f'{feature}_quantile'] = pd.qcut(data[feature], 5, labels=False, duplicates='drop')
            quantile_analysis = data.groupby(f'{feature}_quantile')[target_col].mean()
            
            plt.figure(figsize=(10, 5))
            quantile_analysis.plot(kind='bar', color='skyblue')
            plt.title(f'Rendement Moyen de {target_col} par Quantile de {feature}')
            plt.xlabel(f'Quantile de {feature} (0=Faible, 4=Élevé)')
            plt.ylabel(f'Rendement Moyen à {target_days} jours')
            plt.xticks(rotation=0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.show()
            
        except ValueError as e:
            print(f"Impossible de calculer les quantiles pour {feature} (probablement données non-uniques): {e}")
        except Exception as e:
            print(f"Erreur lors de l'analyse par quantiles : {e}")


if __name__ == "__main__":
    # CORRECTION : Utilisation de start_date et end_date, ajustées pour une période de 20 ans
    analyze_feature_impact(ticker='EURUSD=X', 
                           start_date='2004-12-31', 
                           end_date='2024-12-31', 
                           target_days=3)