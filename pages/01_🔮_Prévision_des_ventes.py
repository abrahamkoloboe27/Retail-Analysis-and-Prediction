# Onglet 9: Prévisions des Ventes
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
st.markdown("<h1 class='header' style='text-align: center; background-color: #ADD8E6;'>🔮 Dashboard Analytics des Ventes - Prévision des Ventes</h1>", unsafe_allow_html=True)

# Charger les données
@st.cache_data
def load_data():
    df_data = pd.read_csv('train.csv', parse_dates=['Date'])
    df_stores = pd.read_csv('store.csv')
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
tab = st.tabs(["🔮 Prévisions des Ventes par Magasin",
    "🌐 Prévisions Globales"])

with tab[0]:
    st.markdown("### 🔮 Prévisions des Ventes par Magasin")

    # Charger le meilleur modèle (XGBoost)
    xgb_pipeline = joblib.load('models/xgboost_pipeline.joblib')

    # Sélectionner les magasins pour la prévision
    store_options = df['Store'].unique()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("🛒 Magasins")
        selected_stores = st.multiselect("Sélectionner les Magasins", options=store_options, default=store_options[:3])

    with c2:
        st.subheader("📅 Date")
        # Sélectionner la période de prévision
        forecast_days = st.slider("Jours à Prévoir", min_value=1, max_value=30, value=20)
        
    with c3:
        st.subheader("📊 Graphique")
        type_of_graph = st.selectbox("Type de Graphique", ["Barres", "Lignes"], index=1)

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for idx, selected_store in enumerate(selected_stores):
        # Filtrer les données pour le magasin sélectionné
        store_df = df[df['Store'] == selected_store].sort_values('Date')
        
        # Préparer les dates futures
        last_date = store_df['Date'].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Store'] = selected_store
        
        # Ajouter les features temporelles
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day'] = future_df['Date'].dt.day
        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
        future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week
        future_df['IsWeekend'] = future_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Récupérer les dernières valeurs pour les autres features
        latest_row = store_df.iloc[-1]
        future_df['Promo'] = latest_row['Promo']
        future_df['CompetitionDistance'] = latest_row['CompetitionDistance']
        future_df['Promo2'] = latest_row['Promo2']
        future_df['StoreType'] = latest_row['StoreType']
        future_df['Assortment'] = latest_row['Assortment']
        future_df['StateHoliday'] = '0'  # Supposons qu'il n'y a pas de jour férié à venir
        
        # Créer des lags pour les ventes
        for lag in range(1, 8):
            future_df[f'Sales_Lag_{lag}'] = store_df['Sales'].shift(lag).iloc[-1]
        
        # Définir les features
        features = ['Store', 'Year', 'Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 
                    'Promo', 'CompetitionDistance', 'Promo2', 'StoreType', 'Assortment', 
                    'StateHoliday'] + [f'Sales_Lag_{lag}' for lag in range(1, 8)]
        
        # Sélectionner les features
        X_future = future_df[features]
        
        # Prédictions
        future_df['Predicted_Sales'] = xgb_pipeline.predict(X_future)
        
        # Ajouter les dernières valeurs connues
        known_df = store_df[['Date', 'Sales']].tail(20)
        known_df.rename(columns={'Sales': 'Known_Sales'}, inplace=True)
        
        color = colors[idx % len(colors)]
        
        if type_of_graph == "Barres":
            fig.add_trace(go.Bar(x=known_df['Date'], y=known_df['Known_Sales'], name=f'Magasin {selected_store} (Connu)', marker_color=color))
            fig.add_trace(go.Bar(x=future_df['Date'], y=future_df['Predicted_Sales'], name=f'Magasin {selected_store} (Prévu)', marker_color=color, opacity=0.6))
        else:
            fig.add_trace(go.Scatter(x=known_df['Date'], y=known_df['Known_Sales'], mode='lines+markers', name=f'Magasin {selected_store} (Connu)', line=dict(color=color)))
            fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted_Sales'], mode='lines+markers', name=f'Magasin {selected_store} (Prévu)', line=dict(color=color, dash='dash')))

    fig.update_layout(title='Prévisions des Ventes', xaxis_title='Date', yaxis_title='Ventes (€)')
    st.plotly_chart(fig, use_container_width=True)

    # Afficher les prévisions dans un tableau
    st.markdown("#### 📄 Tableau des Prévisions")
    # Afficher les magasins choisis dans un expander
    with st.expander("🛒 Magasins Choisis"):
        st.write(f"Magasins sélectionnés: {', '.join(map(str, selected_stores))}")
        st.dataframe(future_df[['Date', 'Predicted_Sales', "Store", "StoreType"]].set_index('Date'),
                     use_container_width=True)

    # Option de téléchargement des prévisions
    csv_forecast = future_df[['Date', 'Predicted_Sales', "Store", "StoreType"]].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les Prévisions",
        data=csv_forecast,
        file_name='sales_forecast.csv',
        mime='text/csv',
    )

# Onglet 10: Prévisions Globales
with tab[1]:
    st.markdown("### 🌐 Prévisions Globales sur Tous les Magasins")
    C1, C2 = st.columns(2)
    with C1:
        # Sélectionner la période de prévision
        forecast_days = st.slider("Nombre de Jours à Prévoir", min_value=1, max_value=30, value=7)
    with C2:
        # Choisir le type de graphique
        chart_type = st.selectbox("Type de graphique", options=["Barres", "Ligne"])
    
    # Préparer les dates futures
    last_date = df['Date'].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
    
    # Préparer un DataFrame pour les prévisions futures
    future_df_list = []
    for store_id in df['Store'].unique():
        store_df = df[df['Store'] == store_id].sort_values('Date')
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['Store'] = store_id
        
        # Ajouter les features temporelles
        future_df['Year'] = future_df['Date'].dt.year
        future_df['Month'] = future_df['Date'].dt.month
        future_df['Day'] = future_df['Date'].dt.day
        future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
        future_df['WeekOfYear'] = future_df['Date'].dt.isocalendar().week
        future_df['IsWeekend'] = future_df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Récupérer les dernières valeurs pour les autres features
        latest_row = store_df.iloc[-1]
        future_df['Promo'] = latest_row['Promo']
        future_df['CompetitionDistance'] = latest_row['CompetitionDistance']
        future_df['Promo2'] = latest_row['Promo2']
        future_df['StoreType'] = latest_row['StoreType']
        future_df['Assortment'] = latest_row['Assortment']
        future_df['StateHoliday'] = '0'  # Supposons qu'il n'y a pas de jour férié à venir
        
        # Créer des lags pour les ventes
        for lag in range(1, 8):
            if len(store_df) >= lag:
                future_df[f'Sales_Lag_{lag}'] = store_df['Sales'].iloc[-lag]
            else:
                future_df[f'Sales_Lag_{lag}'] = store_df['Sales'].mean()  # Utiliser la moyenne si pas assez de données
        
        future_df_list.append(future_df)
    
    # Concaténer toutes les prévisions futures
    future_df_all = pd.concat(future_df_list, ignore_index=True)
    
    # Sélectionner les features
    X_future = future_df_all[features]
    
    # Prédictions
    future_df_all['Predicted_Sales'] = xgb_pipeline.predict(X_future)
    
    # Agréger les prévisions par date
    sales_forecast = future_df_all.groupby('Date')['Predicted_Sales'].sum().reset_index()
    
    # Obtenir les ventes passées (connues)
    known_sales = df.groupby('Date')['Sales'].sum().reset_index().tail(10)
    
    # Afficher les indicateurs clés
    total_predicted_sales = sales_forecast['Predicted_Sales'].sum()
    average_daily_sales = sales_forecast['Predicted_Sales'].mean()
    average_sales_per_store = future_df_all.groupby('Store')['Predicted_Sales'].sum().mean()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box" style="background-color: #FFDDC1;">
            <h3>💰 Ventes Totales Prévisionnelles</h3>
            <p class="big-font">{total_predicted_sales:,.2f} €</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box" style="background-color: #C1FFD7;">
            <h3>📆 Vente Moyenne Quotidienne</h3>
            <p class="big-font">{average_daily_sales:,.2f} €</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box" style="background-color: #C1D4FF;">
            <h3>🏬 Vente Moyenne par Magasin</h3>
            <p class="big-font">{average_sales_per_store:,.2f} €</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Visualiser les prévisions
    if chart_type == "Ligne":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=known_sales['Date'], y=known_sales['Sales'], mode='lines+markers', name='Ventes Connues', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=sales_forecast['Date'], y=sales_forecast['Predicted_Sales'], mode='lines+markers', name='Ventes Prévisionnelles', line=dict(color='red', dash='dash')))
        fig.update_layout(title='Prévisions des Ventes Totales', xaxis_title='Date', yaxis_title='Ventes (€)')
    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=known_sales['Date'], y=known_sales['Sales'], name='Ventes Connues', marker_color='blue'))
        fig.add_trace(go.Bar(x=sales_forecast['Date'], y=sales_forecast['Predicted_Sales'], name='Ventes Prévisionnelles', marker_color='red', opacity=0.6))
        fig.update_layout(title='Prévisions des Ventes Totales', xaxis_title='Date', yaxis_title='Ventes (€)')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Afficher les prévisions dans un tableau
    with st.expander("#### 📄 Tableau des Prévisions Totales"):
        st.dataframe(sales_forecast.set_index('Date'), use_container_width=True)
    
    # Option de téléchargement des prévisions
    csv_forecast = sales_forecast.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger les Prévisions Totales",
        data=csv_forecast,
        file_name='sales_forecast_total.csv',
        mime='text/csv',
    )

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
    <p class='footer'>💡 Dashboard créé avec ❤️ par <a href='https://github.com/AbrahamKOLOBOE' target='_blank'>Abraham KOLOBOE</a></p>
    """,
    unsafe_allow_html=True
)
