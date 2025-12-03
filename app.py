import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm  # Nécessaire pour gérer les modèles statsmodels
import streamlit as st

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Projet Pricing Auto",
    page_icon="🚗",
    layout="wide"
)


# --- CHARGEMENT DU MODÈLE ET DES COLONNES (MISE À JOUR) ---
@st.cache_resource
def load_assets():
    try:
        # 1. Modèle Initial (Sklearn)
        initial_model = joblib.load('model_initial.pkl')
        initial_columns = joblib.load('cols_initial.pkl')

        # 2. Modèle Itératif (Statsmodels)
        iterative_model = joblib.load('model_iteratif.pkl')
        iterative_columns = joblib.load('cols_iteratif.pkl')

        # 3. Modèle Robuste (Statsmodels)
        robust_model = joblib.load('model_robuste.pkl')
        robust_columns = joblib.load('cols_robuste.pkl')

        return {
            'initial': (initial_model, initial_columns),
            'iterative': (iterative_model, iterative_columns),
            'robust': (robust_model, robust_columns)
        }
    except FileNotFoundError as e:
        # On retourne l'erreur pour aider au débogage si un fichier manque
        return f"Erreur de fichier : {e}"


# Chargement des ressources
models_data = load_assets()

# --- CSS PERSONNALISÉ (STYLE PRO) ---
st.markdown("""
 <style>    
    /* === CONFIGURATION GLOBALE === */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e6f1ff;
    }

    /* Conteneur principal */
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* === TITRES === */
    h1 {
        color: #ff6b35 !important;
        font-weight: 800 !important;
        font-size: 3.5rem !important;
        text-align: center;
        margin-bottom: 1rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        letter-spacing: -1px;
    }

    h2 {
        color: #0ea5e9 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.5rem !important;
        border-left: 5px solid #0ea5e9;
        padding-left: 15px;
    }

    /* === BOUTONS === */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 2rem !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 8px 20px rgba(194, 65, 12, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
        transform: translateY(-2px) !important;
    }

    /* === CARTES MÉTRIQUES === */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }

    [data-testid="stMetricValue"] {
        color: #fdc830 !important;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%) !important;
        border-right: 2px solid rgba(255, 107, 53, 0.3);
    }
 </style>
""", unsafe_allow_html=True)

# --- SIDEBAR (NAVIGATION) ---
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png",
    width=50)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Menu", ["Accueil & Contexte", "Approche Statistique & Modèle", "Analyse Visuelle",
                                 "Prédiction Live (Démo)", "Recommandations & Conclusion"])

st.sidebar.markdown("---")
st.sidebar.info("Auteur : **FEBON Sitou Daniel**\n\nEncadrant : **M. Soga Para**\n\nUniversité : **ESGIS**")

# =============================================================================
# PAGE 1 : ACCUEIL
# =============================================================================
if page == "Accueil & Contexte":
    st.title("Analyse du Marché Automobile Américain")
    st.markdown(
        "<h3 style='text-align: center; color: #a8b2d1 !important;'>Étude des annonces Craigslist & Modélisation Prédictive</h3>",
        unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown("""
        ### Objectifs de l'étude
        Le marché de l'automobile d'occasion aux États-Unis est vaste et complexe. Ce projet vise à :
        1.  **Nettoyer** un jeu de données réel et bruité (Big Data).
        2.  **Identifier** les déterminants économiques du prix.
        3.  **Construire** un modèle mathématique robuste pour estimer la valeur d'un véhicule.

        ###  Données Sources
        * **Origine :** Craigslist (Kaggle)
        * **Volume Initial :** 426 880 annonces
        * **Complexité :** Données sales (prix à 0$, valeurs manquantes, doublons).
        """)

    with col2:
        st.image("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?q=80&w=1000&auto=format&fit=crop",
                 caption="Marché US")

    # KPI IMPACTANTS
    st.markdown("###  Chiffres Clés du Projet")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Données Brutes", "426 880", "Annonces")
    kpi2.metric("Données Nettoyées", "201 934", "Observations finales")
    kpi3.metric("Variables Significatives", "48 / 52", "P-value < 0.05")
    kpi4.metric("Précision (R²)", "0.625", "Variance Expliquée")

# =============================================================================
# PAGE 2 : APPROCHE STATISTIQUE
# =============================================================================
elif page == "Approche Statistique & Modèle":
    st.title(" Méthodologie & Modélisation")

    # 1. PROCESSUS STATISTIQUE
    st.header("1. Le Processus de Traitement (Pipeline)")
    st.markdown("Transformation de la donnée brute en connaissance exploitable.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("####  1. Nettoyage")
        st.info("Suppression des colonnes vides (>40%), des doublons stricts et des identifiants inutiles.")
    with col2:
        st.markdown("####  2. Filtrage (IQR)")
        st.info("Élimination des outliers : Prix (500-100k$), Année (1990-2025), Km (1k-300k).")
    with col3:
        st.markdown("####  3. Encodage")
        st.info("Transformation des variables catégorielles (One-Hot Encoding) & Imputation médiane.")
    with col4:
        st.markdown("####  4. Modélisation")
        st.info("Régression Linéaire Multiple (OLS) et validation par Train/Test Split (80/20).")

    st.markdown("---")

    # 2. COMPARATIVO DES MODÈLES
    st.header("2. Comparaison des Trois Modèles")
    st.markdown("""
    Nous avons développé trois variantes du modèle pour assurer la robustesse des résultats :
    """)

    # Mise à jour avec les R² mentionnés dans ta dernière exécution
    r2_initial = 0.621  # Modèle complet Sklearn
    r2_iterative = 0.621  # Modèle optimisé
    r2_robust = 0.621  # Modèle robuste HC3

    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        st.metric("1. Initial (Sklearn)", f"{r2_initial:.3f}", "Baseline")
    with col_comp2:
        st.metric("2. Itératif (Optimisé)", f"{r2_iterative:.3f}", "Parcimonieux")
    with col_comp3:
        st.metric("3. Robuste (HC3)", f"{r2_robust:.3f}", "Fiable")

    st.info("""
    **Observation :** Les scores R² sont quasi-identiques, ce qui est une excellente nouvelle. 
    Cela signifie que nous pouvons utiliser le modèle **Itératif** (moins de variables) ou **Robuste** (plus sûr mathématiquement) 
    sans perdre en pouvoir prédictif par rapport au modèle complexe initial.
    """)

    st.markdown("---")

    # 3. INTERPRÉTATION
    st.header("3. Facteurs d'Influence (Modèle Robuste)")
    st.markdown("Variables ayant le plus fort impact (positif ou négatif) sur le prix.")

    col_stat1, col_stat2 = st.columns([1, 2])
    with col_stat1:
        st.metric(label="Variables Analysées", value="48")
        st.success("Toutes les variables restantes sont significatives à 95%.")
    with col_stat2:
        data_coef = {
            'Variable': ['Pickup / Truck', 'Offroad / 4WD', 'Lexus / Luxe', 'Hybride / Électrique (Vieux)',
                         'Haut Kilométrage'],
            'Tendance': ['Hausse (+++)', 'Hausse (++)', 'Hausse (++)', 'Baisse (---)', 'Baisse (--)'],
            'Interprétation': [
                'Les utilitaires gardent une valeur très élevée.',
                'La capacité tout-terrain est une prime majeure.',
                'Marques fiables et premium.',
                'Forte décote due aux inquiétudes sur les batteries.',
                'Dépréciation classique linéaire.'
            ]
        }
        st.table(pd.DataFrame(data_coef))

# =============================================================================
# PAGE 3 : ANALYSE VISUELLE
# =============================================================================
elif page == "Analyse Visuelle":
    st.title("Exploration Visuelle des Données")

    tabs = st.tabs(["Dépréciation", "Marques & Luxe", "Carburant", "Validation Modèle"])

    with tabs[0]:
        st.subheader("Impact de l'Usure (Km et Âge)")
        c1, c2 = st.columns(2)
        with c1:
            # Assurez-vous que ces images existent dans le dossier
            st.image("graph2_price_vs_odometer.png", caption="Prix vs Kilométrage")
        with c2:
            st.image("graph3_price_vs_age.png", caption="Prix vs Âge")

    with tabs[1]:
        st.subheader("Positionnement des Marques")
        st.write("Distribution des prix pour les marques principales.")
        # Placeholder si l'image n'est pas dispo, sinon mettre le bon nom de fichier
        st.info("Voir le graphique 'Boxplot Manufacturer' généré dans le notebook.")

    with tabs[2]:
        st.subheader("Prix par Motorisation")
        st.image("graph4_boxplot_fuel.png", caption="Prix par Carburant")

    with tabs[3]:
        st.subheader("Diagnostic des Résidus")
        st.image("evaluation_finale_residus.png", caption="Analyse des Résidus")

# =============================================================================
# PAGE 4 : DÉMO LIVE
# =============================================================================
elif page == "Prédiction Live (Démo)":
    st.title("Simulateur de Prix IA")
    st.markdown("Utilisez l'un des trois modèles pour estimer la valeur.")

    if isinstance(models_data, str):  # Gestion d'erreur si fichiers manquants
        st.error(f"⚠️ {models_data}")
        st.warning("Veuillez lancer le script `save_models_final.py` dans votre notebook.")
    else:
        with st.container():
            st.markdown("<div class='info-box'>Paramètres du véhicule</div>", unsafe_allow_html=True)

            # SÉLECTEUR DE MODÈLE
            model_choice = st.selectbox(
                "🧠 Choisissez le Cerveau (Modèle) :",
                options=['initial', 'iterative', 'robust'],
                format_func=lambda x: {
                    'initial': '1. Modèle Initial (Sklearn - Standard)',
                    'iterative': '2. Modèle Itératif (Optimisé - Selectif)',
                    'robust': '3. Modèle Robuste (HC3 - Fiable)'
                }[x],
                index=0
            )

            selected_model, selected_columns = models_data[model_choice]

            with st.form("pred_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    year = st.number_input("Année", 1990, 2025, 2018)
                    odometer = st.number_input("Kilométrage (Miles)", 0, 300000, 45000, step=1000)
                with c2:
                    brands = sorted(
                        ['ford', 'chevrolet', 'toyota', 'honda', 'nissan', 'jeep', 'ram', 'gmc', 'bmw', 'dodge',
                         'mercedes-benz', 'hyundai', 'subaru', 'volkswagen', 'kia', 'lexus', 'audi', 'cadillac',
                         'acura', 'buick', 'other'])
                    manufacturer = st.selectbox("Marque", brands, index=2)
                    types = sorted(
                        ['sedan', 'SUV', 'pickup', 'truck', 'coupe', 'hatchback', 'convertible', 'van', 'wagon',
                         'other'])
                    type_vehicule = st.selectbox("Carrosserie", types, index=1)
                with c3:
                    fuel = st.selectbox("Carburant", ['gas', 'diesel', 'hybrid', 'electric', 'other'])
                    transmission = st.selectbox("Transmission", ['automatic', 'manual', 'other'])
                    drive = st.selectbox("Roues", ['4wd', 'rwd', 'fwd'])

                submit = st.form_submit_button("💰 CALCULER L'ESTIMATION")

        if submit:
            # 1. Préparation du DataFrame vide avec les colonnes attendues par le modèle chargé
            input_df = pd.DataFrame(columns=selected_columns)
            input_df.loc[0] = 0  # Initialisation à 0

            # 2. Remplissage des variables numériques
            car_age = 2025 - year

            # Vérification si les colonnes existent (au cas où le modèle itératif les a supprimées)
            if 'odometer' in input_df.columns: input_df['odometer'] = odometer
            if 'car_age' in input_df.columns: input_df['car_age'] = car_age

            # 3. Encodage One-Hot manuel
            inputs = {
                'manufacturer': manufacturer, 'fuel': fuel,
                'transmission': transmission, 'drive': drive, 'type': type_vehicule
            }
            for col, val in inputs.items():
                col_name = f"{col}_{val}"
                if col_name in input_df.columns:
                    input_df[col_name] = 1

            # 4. Prédiction (Gestion Sklearn vs Statsmodels)
            try:
                # On détecte si c'est un modèle Statsmodels (qui a une méthode 'predict' mais pas 'fit' comme sklearn Wrapper)
                # Ou plus simple: on regarde le type
                is_statsmodels = 'statsmodels' in str(type(selected_model))

                if is_statsmodels:
                    # Statsmodels nécessite une constante explicitement ajoutée pour la prédiction
                    # On ajoute une colonne 'const' à 1.0 au début ou on utilise add_constant
                    input_with_const = sm.add_constant(input_df, has_constant='add')
                    # Force l'ajout si add_constant ne le fait pas sur une seule ligne sans variance
                    if 'const' not in input_with_const.columns:
                        input_with_const.insert(0, 'const', 1.0)

                    price_pred = selected_model.predict(input_with_const)[0]
                else:
                    # Sklearn
                    price_pred = selected_model.predict(input_df)[0]

                st.markdown("---")
                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    st.metric(label="Estimation Estimée", value=f"{price_pred:,.0f} $")

                with res_col2:
                    if price_pred > 0:
                        st.success(f"Véhicule : {manufacturer.upper()} {type_vehicule} ({year})")
                        st.caption(f"Calculé avec : {model_choice.capitalize()}")
                    else:
                        st.error("Résultat atypique. Vérifiez les entrées.")

            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.info("Détail : Vérifiez que les colonnes du fichier pickle correspondent aux entrées.")

# =============================================================================
# PAGE 5 : CONCLUSION
# =============================================================================
elif page == "Recommandations & Conclusion":
    st.title("Recommandations & Conclusion")

    st.markdown("""
    ### 🎯 Synthèse
    Ce projet démontre qu'il est possible de prédire le prix des véhicules d'occasion avec une **précision satisfaisante (R² ~62%)** malgré la complexité du marché réel.

    ### 💡 Points Clés
    1. **L'importance du nettoyage :** 50% du travail a consisté à traiter les valeurs manquantes et aberrantes.
    2. **Le choix du modèle :** Le modèle robuste (HC3) est préférable pour une mise en production car il est moins sensible aux variations extrêmes de prix.
    3. **Les drivers de valeur :** Le type de véhicule (Pickup/Truck) est le déterminant #1 du prix aux USA, devant la marque.
    """)

    if st.button("Terminer la présentation 🎉"):
        st.balloons()