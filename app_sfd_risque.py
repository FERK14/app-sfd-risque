
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la page
st.set_page_config(page_title="Supervision des SFD", layout="wide")

# Charger les données
df = pd.read_excel("donnees_SFD_simulees.xlsx")

st.title("📊 Supervision des SFD - Risque de Faillite")

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

# Alerte IA
st.subheader("🧠 Analyse de risque (IA)")
if sfd_data['En_risque'] == 1:
    st.error("⚠️ Ce SFD présente un profil à risque. Une attention particulière est recommandée.")
else:
    st.success("✅ Ce SFD est considéré comme stable.")

# Visualisation comparative
st.subheader("📈 Comparaison du ratio de liquidité")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=df['Ratio_liquidite'], y=df.index, orient='h', ax=ax, palette="coolwarm")
ax.axvline(x=100, color='red', linestyle='--')
ax.set_xlabel("Ratio de liquidité (%)")
ax.set_ylabel("SFD")
st.pyplot(fig)
