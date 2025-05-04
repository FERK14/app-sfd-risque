import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document

st.set_page_config(page_title="Supervision IA des SFD", layout="wide")

# Charger les données
df = pd.read_excel("donnees_SFD_simulees.xlsx")

# Préparation des données pour le modèle
X = df.drop(columns=['En_risque'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Entraînement du modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, df['En_risque'])

# Prédictions
df['Prediction_IA'] = model.predict(X_scaled)
df['Probabilité_Risque'] = model.predict_proba(X_scaled)[:, 1]

st.title("📊 Supervision des SFD - IA & Rapports")

# SECTION 1 : Visualisation interactive
st.header("🔍 Analyse individuelle")
sfd_index = st.selectbox("Sélectionnez un SFD", df.index)
sfd_data = df.loc[sfd_index]

st.subheader("📌 Indicateurs financiers")
col1, col2, col3 = st.columns(3)
col1.metric("Encours crédits (FCFA)", f"{sfd_data['Encours_credits']:,}")
col2.metric("Créances douteuses (%)", f"{sfd_data['Creances_douteuses']:.2f}")
col3.metric("Ratio liquidité (%)", f"{sfd_data['Ratio_liquidite']:.2f}")

col4, col5, col6 = st.columns(3)
col4.metric("Ratio solvabilité (%)", f"{sfd_data['Ratio_solvabilite']:.2f}")
col5.metric("Nombre d'agences", int(sfd_data['Nb_agences']))
col6.metric("Rendement actifs (%)", f"{sfd_data['Rendement_actifs']:.2f}")

st.subheader("🧠 Prédiction IA")
proba = sfd_data['Probabilité_Risque']
if sfd_data['Prediction_IA'] == 1:
    st.error(f"⚠️ Risque détecté (probabilité : {proba:.2%})")
else:
    st.success(f"✅ Pas de risque détecté (probabilité : {proba:.2%})")

st.subheader("📈 Comparaison des ratios de liquidité")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=df['Ratio_liquidite'], y=df.index, orient='h', ax=ax, palette="coolwarm")
ax.axvline(x=100, color='red', linestyle='--')
ax.set_xlabel("Ratio de liquidité (%)")
ax.set_ylabel("SFD")
st.pyplot(fig)

# SECTION 2 : Génération de rapport
st.header("📝 Génération de rapport pour les 10 SFD les plus à risque")

if st.button("Générer le rapport Word"):
    top_10 = df.sort_values(by='Probabilité_Risque', ascending=False).head(10)
    doc = Document()
    doc.add_heading('Rapport de supervision - Top 10 SFD à risque', 0)

    for idx, row in top_10.iterrows():
        doc.add_heading(f"SFD {idx+1}", level=1)
        doc.add_paragraph(f"Encours : {row['Encours_credits']:,} FCFA")
        doc.add_paragraph(f"Taux de créances douteuses : {row['Creances_douteuses']:.2f} %")
        doc.add_paragraph(f"Ratio de liquidité : {row['Ratio_liquidite']:.2f} %")
        doc.add_paragraph(f"Ratio de solvabilité : {row['Ratio_solvabilite']:.2f} %")
        doc.add_paragraph(f"Rendement des actifs : {row['Rendement_actifs']:.2f} %")
        doc.add_paragraph(f"Score IA (probabilité de risque) : {row['Probabilité_Risque']:.2%}")
        doc.add_paragraph("📝 Analyse automatisée :")
        doc.add_paragraph("Ce SFD présente un profil à risque selon le modèle d'IA. Un renforcement du contrôle et une inspection approfondie sont recommandés.")

    rapport_path = "rapport_top_10_sfd.docx"
    doc.save(rapport_path)

    with open(rapport_path, "rb") as f:
        st.download_button("📥 Télécharger le rapport Word", f, file_name=rapport_path)
