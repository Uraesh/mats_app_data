import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm  # Ajouté pour gérer les prédictions des modèles statsmodels
import streamlit as st

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Projet Pricing Auto",
    page_icon="🚗",
    layout="wide"
)


# --- CHARGEMENT DU MODÈLE ET DES COLONNES ---
@st.cache_resource
def load_assets():
    try:
        # Chargement du modèle initial (premier modèle de régression linéaire)
        initial_model = joblib.load('linear_regression_model.pkl')
        initial_columns = joblib.load('model_columns.pkl')

        # Chargement du modèle itératif (deuxième modèle après élimination des features non significatives)
        iterative_model = joblib.load(
            'iterative_model.pkl')  # Assurez-vous que ce fichier existe après avoir sauvegardé le modèle itératif
        iterative_columns = joblib.load('iterative_model_columns.pkl')

        # Chargement du modèle robuste (troisième modèle avec covariance HC3 pour gérer l'hétéroscédasticité)
        robust_model = joblib.load(
            'robust_model.pkl')  # Assurez-vous que ce fichier existe après avoir sauvegardé le modèle robuste
        robust_columns = joblib.load('robust_model_columns.pkl')

        return {
            'initial': (initial_model, initial_columns),
            'iterative': (iterative_model, iterative_columns),
            'robust': (robust_model, robust_columns)
        }
    except FileNotFoundError:
        return None


models = load_assets()

# --- CSS PERSONNALISÉ (STYLE PRO) ---
st.markdown("""
 <style>    
    /* === CONFIGURATION GLOBALE === */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #e6f1ff;
    }

    /* Conteneur principal avec padding amélioré */
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

    h3 {
        color: #fb923c !important;
        font-weight: 600 !important;
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
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
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button:hover {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 30px rgba(194, 65, 12, 0.6) !important;
    }

    .stButton>button:active {
        transform: translateY(0px) !important;
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
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.5);
        border-color: rgba(255, 107, 53, 0.3);
    }

    /* Métriques Streamlit natives */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        color: #fdc830 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }

    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #a8b2d1 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stMetricDelta"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    /* === DATAFRAMES & TABLES === */
    [data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        overflow: hidden;
    }

    /* En-têtes de tableau */
    .dataframe thead tr th {
        background: linear-gradient(135deg, #0c4a6e 0%, #075985 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        font-size: 0.9rem !important;
        padding: 15px !important;
        border: none !important;
    }

    .dataframe tbody tr:hover {
        background-color: rgba(14, 165, 233, 0.1) !important;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%) !important;
        border-right: 2px solid rgba(255, 107, 53, 0.3);
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ff6b35 !important;
    }

    /* === SELECTBOX & INPUTS === */
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #e6f1ff !important;
    }

    .stTextInput > div > div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #e6f1ff !important;
    }

    /* === GRAPHIQUES PLOTLY === */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        color: #a8b2d1;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c2410c 0%, #ea580c 100%);
        color: white !important;
        border: none;
    }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #0ea5e9 !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(255, 107, 53, 0.15) 0%, rgba(255, 107, 53, 0.05) 100%) !important;
        border-color: rgba(255, 107, 53, 0.3) !important;
    }

    /* === ALERTS & INFO BOXES === */
    .stAlert {
        background-color: rgba(14, 165, 233, 0.1) !important;
        border-left: 5px solid #0ea5e9 !important;
        border-radius: 8px !important;
        color: #e6f1ff !important;
    }

    /* === SPINNER === */
    .stSpinner > div {
        border-top-color: #ff6b35 !important;
    }

    /* === PROGRESS BAR === */
    .stProgress > div > div {
        background-color: #c2410c !important;
    }

    /* === DIVIDER === */
    hr {
        border-color: rgba(255, 107, 53, 0.3) !important;
        margin: 2rem 0 !important;
    }

    /* === CARDS CUSTOM (À utiliser avec st.container) === */
    .insight-card {
        background: linear-gradient(135deg, rgba(5, 150, 105, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #059669;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .warning-card {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #dc2626;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* === TEXTE === */
    p, li, span {
        color: #a8b2d1 !important;
        line-height: 1.7 !important;
        font-size: 1.05rem !important;
    }

    /* Code blocks */
    code {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #fdc830 !important;
        padding: 2px 8px !important;
        border-radius: 5px !important;
        font-family: 'Courier New', monospace !important;
    }

    /* === ANIMATIONS === */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .metric-card, .insight-card, .warning-card {
        animation: fadeInUp 0.5s ease-out;
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
    kpi3.metric("Variables Significatives", "48 / 48", "P-value < 0.05")
    kpi4.metric("Précision (R²)", "0.6252", "Variance Expliquée")

# =============================================================================
# PAGE 2 : APPROCHE STATISTIQUE (NOUVEAU)
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

    # 2. L'ÉQUATION DU MODÈLE
    st.header("2. L'Équation Mathématique du Prix")
    st.markdown("Le modèle estime le prix ($Y$) comme une combinaison linéaire des caractéristiques ($X$).")

    st.latex(r'''
    \text{Prix} = \beta_0 + \beta_1 \times \text{Kilométrage} + \beta_2 \times \text{Âge} + \sum_{i=3}^{n} \beta_i \times X_i + \epsilon
    ''')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        * **$\beta_0$ (Constante)** : 50 150 $ (Prix de base théorique).
        * **$\beta_1$ (Kilométrage)** : -0.075 $ par mile parcouru.
        * **$\beta_2$ (Âge)** : -853 $ par année d'ancienneté.
        """)
    with c2:
        st.markdown("""
        * **$\beta_i$ (Variables Catégorielles)** : Bonus/Malus selon la Marque, le Carburant, etc.
        * **$\epsilon$ (Erreur)** : Résidu aléatoire (Moyenne = 0).
        """)

    st.markdown("---")

    # Ajout : Section pour comparer les 3 modèles
    st.header("3. Comparaison des Trois Modèles")
    st.markdown("""
    Nous avons développé trois variantes du modèle de régression linéaire :
    - **Modèle Initial** : Version de base avec toutes les features.
    - **Modèle Itératif** : Après élimination itérative des features non significatives (p-value > 0.05).
    - **Modèle Robuste** : Version finale avec covariance robuste (HC3) pour gérer l'hétéroscédasticité.
    """)

    # Exemples de R² (remplacez par des valeurs réelles si vous avez accès aux summaries des modèles)
    r2_initial = 0.6252737153637336  # Valeur du modèle initial
    r2_iterative =  0.6252698761044901  # Valeur après itération (légèrement ajustée)
    r2_robust = 0.6252592290034307  # Valeur du modèle robuste

    # Affichage des métriques de comparaison
    col_comp1, col_comp2, col_comp3 = st.columns(3)
    with col_comp1:
        st.metric("R² Modèle Initial", f"{r2_initial:.3f}")
    with col_comp2:
        st.metric("R² Modèle Itératif", f"{r2_iterative:.3f}")
    with col_comp3:
        st.metric("R² Modèle Robuste", f"{r2_robust:.3f}")

    st.info("""
    Le modèle robuste est sélectionné comme meilleur car il gère l'hétéroscédasticité tout en maintenant un R² élevé.
    Différences mineures : L'itération supprime du bruit, et la robustesse ajuste les erreurs standard pour plus de fiabilité.
    """)

    st.markdown("---")

    # 3. TABLEAU SYNTHÉTIQUE (adapté pour le modèle robuste v3)
    st.header("4. Significativité des Variables (Modèle Robuste v3)")
    st.markdown(
        "Analyse approfondie des P-values (Test Z robuste) pour valider la pertinence des variables dans ce modèle raffiné. Après suppression des features non significatives antérieures, toutes les variables restantes contribuent de manière statistiquement significative (p < 0.05). Le modèle est plus parcimonieux avec 48 variables explicatives.")

    col_stat1, col_stat2 = st.columns([1, 2])

    with col_stat1:
        st.metric(label="Variables Explicatives Totales", value="48")
        st.metric(label="Variables Significatives (p < 0.05)", value="48", delta="100% Pertinence")
        st.success(
            "Aucune variable non significative dans ce modèle raffiné. Toutes contribuent au prix de manière fiable.")

    with col_stat2:
        st.markdown("#### 🏆 Top 5 des Facteurs d'Influence (Coefficients Positifs)")
        # DataFrame mis à jour avec les nouveaux coefficients (arrondis)
        data_coef = {
            'Variable': ['type_pickup', 'type_offroad', 'type_truck', 'manufacturer_lexus', 'type_convertible'],
            'Impact sur le Prix': ['+5828 $', '+5356 $', '+5135 $', '+4358 $', '+3501 $'],
            'Interprétation': ['Forte valorisation utilitaire (e.g., trucks robustes)',
                               'Capacité de franchissement et aventure',
                               'Usage professionnel et capacité de charge',
                               'Prime à la fiabilité et au luxe japonais',
                               'Véhicule plaisir et style décapotable']
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
            st.image("graph2_price_vs_odometer.png", caption="Prix vs Kilométrage", use_column_width=True)
            st.caption("On observe une décroissance convexe : forte perte de valeur au début, puis stabilisation.")
        with c2:
            st.image("graph3_price_vs_age.png", caption="Prix vs Âge", use_column_width=True)
            st.caption("Corrélation négative nette. Les véhicules récents ont une variance de prix plus élevée.")

    with tabs[1]:
        st.subheader("Positionnement des Marques")
        st.image("graph3_price_vs_age.png", caption="Distribution des prix par Marque",
                 use_column_width=True)  # Assure toi d'avoir cette image ou graph5_boxplot_drive.png
        st.write("Lexus et les marques de Pick-ups se distinguent nettement.")

    with tabs[2]:
        st.subheader("Le Diesel : Une exception américaine")
        st.image("graph4_boxplot_fuel.png", caption="Prix par Carburant", use_column_width=True)
        st.success(
            "Insight Business : Contrairement à l'Europe, le Diesel (associé aux gros Trucks) est la motorisation la plus chère, loin devant l'essence et l'hybride.")

    with tabs[3]:
        st.subheader("Diagnostic du Modèle")
        st.image("evaluation_finale_residus.png", caption="Analyse des Résidus", use_column_width=True)
        st.info(
            "La distribution normale des résidus (courbe en cloche) valide la robustesse statistique du modèle, malgré une légère hétéroscédasticité sur les prix élevés.")

# =============================================================================
# PAGE 4 : DÉMO LIVE
# =============================================================================
elif page == "Prédiction Live (Démo)":
    st.title("Simulateur de Prix IA")
    st.markdown("Utilisez le modèle en temps réel pour estimer la valeur d'un véhicule.")

    if models is None:
        st.error("Modèles introuvables. Veuillez exécuter les scripts de sauvegarde pour générer les .pkl.")
    else:
        with st.container():
            st.markdown("<div class='info-box'>Configurez les paramètres ci-dessous pour lancer la prédiction.</div>",
                        unsafe_allow_html=True)

            # Ajout : Sélection du modèle à utiliser pour la prédiction
            model_choice = st.selectbox(
                "Choisissez le modèle à utiliser :",
                options=['initial', 'iterative', 'robust'],
                format_func=lambda x: {
                    'initial': 'Modèle Initial (Base)',
                    'iterative': 'Modèle Itératif (Features Sélectionnées)',
                    'robust': 'Modèle Robuste (HC3)'
                }[x],
                index=2  # Robuste par défaut
            )
            selected_model, selected_columns = models[model_choice]

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
                    manufacturer = st.selectbox("Marque", brands, index=2)  # Toyota par défaut
                    types = sorted(
                        ['sedan', 'SUV', 'pickup', 'truck', 'coupe', 'hatchback', 'convertible', 'van', 'wagon',
                         'other'])
                    type_vehicule = st.selectbox("Carrosserie", types, index=0)
                with c3:
                    fuel = st.selectbox("Carburant", ['gas', 'diesel', 'hybrid', 'electric', 'other'])
                    transmission = st.selectbox("Transmission", ['automatic', 'manual', 'other'])
                    drive = st.selectbox("Roues", ['4wd', 'rwd', 'fwd'])

                submit = st.form_submit_button("CALCULER LA VALEUR")

        if submit:
            # Préparation des données
            car_age = 2025 - year
            input_df = pd.DataFrame(np.zeros((1, len(selected_columns))), columns=selected_columns)
            input_df['odometer'] = odometer
            input_df['car_age'] = car_age

            # Encodage One-Hot manuel
            inputs = {
                'manufacturer': manufacturer, 'fuel': fuel,
                'transmission': transmission, 'drive': drive, 'type': type_vehicule
            }
            for col, val in inputs.items():
                col_name = f"{col}_{val}"
                if col_name in input_df.columns:
                    input_df[col_name] = 1

            # Prédiction avec le modèle sélectionné, en gérant les différences entre sklearn et statsmodels
            if hasattr(selected_model, 'predict') and 'statsmodels' in str(type(selected_model)):
                # Pour statsmodels OLS : Ajouter la constante et utiliser predict(exog)
                exog = sm.add_constant(input_df, has_constant='add')
                price_pred = selected_model.predict(exog)[0]
            else:
                # Pour sklearn LinearRegression : Pas de constante supplémentaire
                price_pred = selected_model.predict(input_df)[0]

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])

            with res_col1:
                st.metric(label="Estimation", value=f"{price_pred:,.0f} $")

            with res_col2:
                if price_pred > 0:
                    st.success(f"Véhicule analysé : {manufacturer.upper()} {type_vehicule} de {year}")
                    if fuel == 'diesel': st.write(
                        "💡 **Bonus Diesel :** Ce véhicule bénéficie d'une forte cote grâce à sa motorisation.")
                    if type_vehicule in ['pickup', 'truck']: st.write(
                        "💡 **Bonus Utilitaire :** Les véhicules de travail gardent une excellente valeur.")
                    if car_age > 15: st.warning("**Attention :** L'âge avancé pèse fortement sur l'estimation.")
                else:
                    st.error(
                        "Résultat atypique (négatif). Les caractéristiques saisies sortent des standards du modèle.")

# =============================================================================
# PAGE 5 : RECOMMANDATIONS & CONCLUSION
# =============================================================================
elif page == "Recommandations & Conclusion":
    st.title("Recommandations Stratégiques")
    st.markdown("Synthèse opérationnelle pour les acteurs du marché.")

    rec_tabs = st.tabs(["Pour les Vendeurs", "Pour les Acheteurs", "Pour les Plateformes"])

    with rec_tabs[0]:
        st.markdown("""
        <div class='success-box'>
        <h4>Conseils aux Vendeurs</h4>
        <ul>
            <li><strong>Mise en avant :</strong> Valorisez explicitement le faible kilométrage et l'entretien récent.</li>
            <li><strong>Transparence :</strong> Complétez toutes les infos (Carburant, Transmission). Les annonces "Incomplètes" subissent une décote statistique.</li>
            <li><strong>Haut de Gamme :</strong> Pour les véhicules > 40k$, ajoutez des photos détaillées, le modèle sous-estime parfois leur valeur réelle.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_tabs[1]:
        st.markdown("""
        <div class='info-box'>
        <h4>Conseils aux Acheteurs</h4>
        <ul>
            <li><strong>Détecteur de Bonnes Affaires :</strong> Si le prix affiché est nettement inférieur à notre prédiction IA, c'est une opportunité (ou une arnaque à vérifier).</li>
            <li><strong>Vigilance :</strong> Méfiance si le prix est > 50% sous l'estimation (risque d'accident caché).</li>
            <li><strong>Astuce :</strong> Les berlines essence décotent plus vite que les SUV/Pickups. Bon plan pour petit budget.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with rec_tabs[2]:
        st.markdown("""
        <div class='metric-card' style='text-align: left;'>
        <h4>Recommandations pour l'Entreprise (Craigslist)</h4>
        <ol>
            <li><strong>Intégration API :</strong> Proposer ce modèle en temps réel lors de la création d'annonce pour suggérer un "Prix Juste".</li>
            <li><strong>Détection de Fraude :</strong> Flagger automatiquement les annonces dont le prix dévie de ±40% de la prédiction.</li>
            <li><strong>Segmentation :</strong> Développer un sous-modèle spécifique pour les véhicules de collection (>30 ans) et Luxe.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Conclusion Générale")
    st.write("""
    Ce projet a permis de transformer des données brutes en un outil d'aide à la décision fiable. 
    Avec **62% de variance expliquée** et une validation statistique robuste (**49 variables significatives**), 
    le modèle démontre que le marché US suit des règles économiques logiques : **L'âge déprécie, l'utilité valorise.**
    """)

    if st.button("Célébrer la fin du projet"):
        st.balloons()
