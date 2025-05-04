import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Supervision des SFD", layout="wide")

# Charger les données
df = pd.read_excel("donnees_SFD_simulees.xlsx")

# Préparation des données (mêmes étapes que pour l'entraînement du modèle)
X = df.drop(columns=['En_risque'])  # On ne triche pas : on ignore la colonne cible réelle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraînement du modèle Random Forest sur les données (à la volée ici)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, df['En_risque'])  # Entraîné avec les vraies étiquettes

# Prédire sur l'ensemble du dataset
df['Prediction_IA'] = model.predict(X_scaled)
df['Probabilité_Risque'] = model.predict_proba(X_scaled)[:, 1]  # Colonne des probabilités

st.title("📊 Supervision des SFD - Risque de Faillite (IA)")

# Sélection du SFD
sfd_index = st.selectbox("Sélectionnez un SFD", df.index)
sfd_data = df.loc[sfd_index]

# Affichage des indicateurs financiers
st.subheader("🔍 Détails de l'institution sélectionnée")
col1, col2, col3 = st.columns(3)
col1.metric("Encours crédits (FCFA)", f"{sfd_data['Encours_credits']:,}")
col2.metric("Créances douteuses (%)", f"{sfd_data['Creances_douteuses']:.2f}")
col3.metric("Ratio liquidité (%)", f"{sfd_data['Ratio_liquidite']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Ratio solvabilité (%)", f"{sfd_data['Ratio_solvabilite']:.2f}")
col5.metric("Nombre d'agences", int(sfd_data['Nb_agences']))
col6.metric("Rendement actifs (%)", f"{sfd_data['Rendement_actifs']:.2f}")

# Affichage de la prédiction IA
st.subheader("🧠 Prédiction du modèle IA")
proba = sfd_data['Probabilité_Risque']
if sfd_data['Prediction_IA'] == 1:
    st.error(f"⚠️ Risque détecté avec une probabilité de {proba:.2%}")
else:
    st.success(f"✅ Pas de risque détecté (probabilité : {proba:.2%})")

# Visualisation comparative
st.subheader("📈 Comparaison des ratios de liquidité")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=df['Ratio_liquidite'], y=df.index, orient='h', ax=ax, palette="coolwarm")
ax.axvline(x=100, color='red', linestyle='--')
ax.set_xlabel("Ratio de liquidité (%)")
ax.set_ylabel("SFD")
st.pyplot(fig)
