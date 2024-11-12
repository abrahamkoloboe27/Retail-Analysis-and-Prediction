# dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import plotly.graph_objects as go

# Configuration de la page
st.set_page_config(page_title="📊 Dashboard Analytics des Ventes", layout="wide")

# Styles personnalisés avec CSS
def local_css():
    st.markdown(
        """
        <style>
        .big-font {
            font-size:20px !important;
        }
        .header {
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 5px;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border-radius: 5px;
            padding: 10px 24px;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

local_css()

# Titre du Dashboard avec emoji
st.markdown("<h1 class='header' style='text-align: center; background-color: #ADD7E6;'>🔮 Dashboard Analytics des Ventes </h1>", unsafe_allow_html=True)

# Charger les données
@st.cache_data
def load_data():
    df_data = pd.read_csv('data/train.csv', parse_dates=['Date'])
    df_stores = pd.read_csv('data/store.csv')
    df = pd.merge(df_data, df_stores, on='Store', how='left')
    return df

df = load_data()

# Traitement des données
def preprocess_data(df):
    # Conversion des colonnes de dates Promo2
    df['CompetitionOpenSinceMonth'] = pd.to_numeric(df['CompetitionOpenSinceMonth'], errors='coerce')
    df['CompetitionOpenSinceYear'] = pd.to_numeric(df['CompetitionOpenSinceYear'], errors='coerce')
    df['Promo2SinceYear'] = pd.to_numeric(df['Promo2SinceYear'], errors='coerce')
    df['Promo2SinceWeek'] = pd.to_numeric(df['Promo2SinceWeek'], errors='coerce')
    
    # Remplacer les valeurs manquantes
    df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
    df['PromoInterval'].fillna('None', inplace=True)
    
    # Ajouter des features temporelles
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # S'assurer que 'StateHoliday' est une chaîne
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    # Création de features de décalage
    for i in range(1, 8):
        df[f'Sales_lag_{i}'] = df.groupby('Store')['Sales'].shift(i)
    
    # Supprimer les lignes avec des valeurs manquantes après le décalage
    df.dropna(inplace=True)
    
    return df

df = preprocess_data(df)

# Sidebar pour les filtres avec style personnalisé
st.sidebar.header("🔍 Filtres")

# Filtre par magasin
store_options = df['Store'].unique()
if st.sidebar.toggle("Sélectionner tous les magasins", True):
    selected_store = store_options
else:
    selected_store = st.sidebar.multiselect("Sélectionner le(s) magasin(s)", options=store_options, default=store_options[:3])

# Filtre par date
min_date = df['Date'].min()
max_date = df['Date'].max()
if st.sidebar.toggle("Sélectionner toutes les dates", True):
    selected_date = [min_date, max_date]
else:
    selected_date = st.sidebar.date_input("Sélectionner la plage de dates", [min_date, max_date])

# Filtre par type de magasin
store_type_options = df['StoreType'].unique()
if st.sidebar.toggle("Sélectionner tous les types de magasin", True):
    selected_store_type = store_type_options
else:
    selected_store_type = st.sidebar.multiselect("Sélectionner le(s) type(s) de magasin", options=store_type_options, default=store_type_options)

# Filtre par assortiment
assortment_options = df['Assortment'].unique()
if st.sidebar.toggle("Sélectionner tous les niveaux d'assortiment", True):
    selected_assortment = assortment_options
else:
    selected_assortment = st.sidebar.multiselect("Sélectionner le(s) niveau(x) d'assortiment", options=assortment_options, default=assortment_options)

# Appliquer les filtres
mask = (
    df['Store'].isin(selected_store) &
    (df['Date'] >= pd.to_datetime(selected_date[0])) &
    (df['Date'] <= pd.to_datetime(selected_date[1])) &
    df['StoreType'].isin(selected_store_type) &
    df['Assortment'].isin(selected_assortment)
)

filtered_df = df.loc[mask]

# Création des onglets
tabs = st.tabs(["📋 Vue d'ensemble", "📅 Ventes au Fil du Temps", "🏪 Analyse par Type & Assortiment", 
                "🎁 Promotions & Jours Fériés", "📍 Distance à la Compétition", "👥 Analyse des Clients", 
                "📊 Corrélations", "📄 Aperçu des Données"])

# Onglet 1: Vue d'ensemble
with tabs[0]:
    st.markdown("### 🌟 Statistiques Clés")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sales = filtered_df['Sales'].sum()
        st.markdown(f"""
        <div class="metric-box" style="background-color: #FFDDC1;">
            <h3>💰 Total des Ventes</h3>
            <p class="big-font">{total_sales:,.2f} €</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        average_sales = filtered_df['Sales'].mean()
        st.markdown(f"""
        <div class="metric-box" style="background-color: #C1E1FF;">
            <h3>📈 Vente Moyenne</h3>
            <p class="big-font">{average_sales:,.2f} €</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_customers = filtered_df['Customers'].sum()
        st.markdown(f"""
        <div class="metric-box" style="background-color: #C1FFC1;">
            <h3>👥 Total des Clients</h3>
            <p class="big-font">{total_customers:,}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        average_customers = filtered_df['Customers'].mean()
        st.markdown(f"""
        <div class="metric-box" style="background-color: #FFD1DC;">
            <h3>🔢 Nombre Moyen de Clients</h3>
            <p class="big-font">{average_customers:,.2f}</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Bouton de téléchargement des données filtrées
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les Données Filtrées",
        data=csv,
        file_name='filtered_sales_data.csv',
        mime='text/csv',
    )

# Onglet 2: Ventes au Fil du Temps
with tabs[1]:
    st.markdown("### 📅 Ventes au Fil du Temps")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Ventes Quotidiennes")
        sales_daily = filtered_df.groupby('Date')['Sales'].sum().reset_index()
        fig1 = px.line(sales_daily, x='Date', y='Sales', title='Ventes Quotidiennes', 
                       labels={'Sales': 'Ventes (€)', 'Date': 'Date'})
        fig1.update_layout(hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📆 Ventes Mensuelles")
        sales_monthly = filtered_df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
        sales_monthly['Date'] = pd.to_datetime(sales_monthly[['Year', 'Month']].assign(DAY=1))
        fig2 = px.line(sales_monthly, x='Date', y='Sales', title='Ventes Mensuelles', 
                       labels={'Sales': 'Ventes (€)', 'Date': 'Date'})
        fig2.update_layout(hovermode="x unified")
        st.plotly_chart(fig2, use_container_width=True)

# Onglet 3: Analyse par Type de Magasin et Assortiment
with tabs[2]:
    st.markdown("### 🏪 Analyse par Type de Magasin et Assortiment")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Ventes par Type de Magasin")
        sales_by_store_type = filtered_df.groupby('StoreType')['Sales'].sum().reset_index()
        fig3 = px.bar(sales_by_store_type, x='StoreType', y='Sales', 
                      title='Ventes par Type de Magasin', 
                      labels={'Sales': 'Ventes (€)', 'StoreType': 'Type de Magasin'},
                      color='StoreType')
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("📦 Ventes par Assortiment")
        sales_by_assortment = filtered_df.groupby('Assortment')['Sales'].sum().reset_index()
        fig4 = px.bar(sales_by_assortment, x='Assortment', y='Sales', 
                      title='Ventes par Assortiment', 
                      labels={'Sales': 'Ventes (€)', 'Assortment': 'Assortiment'},
                      color='Assortment')
        st.plotly_chart(fig4, use_container_width=True)

# Onglet 4: Impact des Promotions et Jours Fériés
with tabs[3]:
    st.markdown("### 🎁 Impact des Promotions et Jours Fériés")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Ventes avec vs sans Promotion")
        promo_sales = filtered_df.groupby('Promo')['Sales'].sum().reset_index()
        promo_sales['Promo'] = promo_sales['Promo'].map({0: 'Sans Promo', 1: 'Avec Promo'})
        fig5 = px.pie(promo_sales, names='Promo', values='Sales', 
                     title='Ventes avec vs sans Promotion', 
                     color='Promo',
                     color_discrete_map={'Sans Promo':'#636EFA', 'Avec Promo':'#EF553B'})
        st.plotly_chart(fig5, use_container_width=True)
    
    with col2:
        st.subheader("📅 Ventes pendant les Jours Fériés")
        holiday_sales = filtered_df.groupby('StateHoliday')['Sales'].sum().reset_index()
        holiday_sales['StateHoliday'] = holiday_sales['StateHoliday'].replace({'0': 'None', 'a': 'Public Holiday', 'b': 'Easter Holiday', 'c': 'Christmas'})
        fig6 = px.pie(holiday_sales, names='StateHoliday', values='Sales', 
                     title='Ventes pendant les Jours Fériés', 
                     color='StateHoliday')
        st.plotly_chart(fig6, use_container_width=True)

# Onglet 5: Distance à la Compétition
with tabs[4]:
    st.markdown("### 📍 Distance à la Compétition")
    st.subheader("📏 Distribution de la Distance à la Compétition")
    fig7 = px.histogram(filtered_df, x='CompetitionDistance', nbins=50, 
                       title='Distribution de la Distance à la Compétition', 
                       labels={'CompetitionDistance': 'Distance (m)'}, 
                       color_discrete_sequence=['#FFA15A'])
    st.plotly_chart(fig7, use_container_width=True)

# Onglet 6: Analyse des Clients
with tabs[5]:
    st.markdown("### 👥 Analyse des Clients")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Nombre de Clients Quotidiens")
        customers_daily = filtered_df.groupby('Date')['Customers'].sum().reset_index()
        fig8 = px.line(customers_daily, x='Date', y='Customers', title='Nombre de Clients Quotidiens', 
                       labels={'Customers': 'Nombre de Clients', 'Date': 'Date'})
        fig8.update_layout(hovermode="x unified")
        st.plotly_chart(fig8, use_container_width=True)
    
    with col2:
        st.subheader("🏬 Nombre de Clients par Magasin")
        n_to_show = st.slider("Nombre de Magasins à Afficher", 1, len(filtered_df["Store"].unique()), 50)
        customers_store = filtered_df.groupby('Store')['Customers'].sum().reset_index().nlargest(n_to_show, 'Customers')
        fig9 = px.bar(customers_store, x='Store', y='Customers', 
                     title='Nombre de Clients par Magasin', 
                     labels={'Customers': 'Nombre de Clients', 'Store': 'Magasin'},
                     color='Customers',
                     color_continuous_scale='Viridis')
        st.plotly_chart(fig9, use_container_width=True)

# Onglet 7: Corrélations
with tabs[6]:
    st.markdown("### 📊 Corrélations")
    st.subheader("📈 Matrice de Corrélation")
    corr_columns = ['Sales', 'Customers', 'CompetitionDistance', 'Promo', 'IsWeekend']
    corr = filtered_df[corr_columns].corr().round(4)
    fig10 = px.imshow(corr, text_auto=True, 
                      title='Matrice de Corrélation', 
                      color_continuous_scale='Viridis',
                      width=1200, height=600)
    st.plotly_chart(fig10, use_container_width=True)

# Onglet 8: Aperçu des Données
with tabs[7]:
    st.markdown("### 📄 Aperçu des Données Filtrées")
    st.dataframe(filtered_df.head(100))

    # Bouton pour afficher plus de données
    if st.button('🔍 Afficher plus de données'):
        st.dataframe(filtered_df)


# Footer avec emojis et style
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        color: #4CAF50;
        font-size: 16px;
        margin-top: 20px;
    }
    .footer a {
        color: #4CAF50;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <p class='footer'>💡 Dashboard créé avec ❤️ par <a href='https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning/' target='_blank'>Abraham KOLOBOE</a></p>
    """,
    unsafe_allow_html=True
)
