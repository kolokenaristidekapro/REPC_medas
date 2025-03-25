import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Génération des données
np.random.seed(42)
X = np.random.rand(100, 1) * 10
Y = 3 * X + np.random.randn(100, 1) * 4

# Entraînement du modèle
model = LinearRegression()
model.fit(X, Y)

# Prédiction sur de nouvelles valeurs
X_new = np.array([[5], [7], [9]])  # Nouvelles valeurs pour prédire Y
Y_pred = model.predict(X_new)

# Enregistrement des prédictions dans un fichier CSV
df = pd.DataFrame({"X": X_new.flatten(), "Y_prédite": Y_pred.flatten()})
df.to_csv("predictions.csv", index=False)

print("Prédictions enregistrées dans predictions.csv")

