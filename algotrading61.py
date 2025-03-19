import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Title
st.title("Sensex Stock Analysis and Recommendations")
st.write("Understand Sensex and invest in it as a whole")

# Define the list of SENSEX stock tickers
stocks = [
    'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJFINANCE.NS',
    'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'HDFCBANK.NS',
    'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS',
    'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS',
    'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS',
    'POWERGRID.NS', 'RELIANCE.NS', 'SBIN.NS', 'SUNPHARMA.NS',
    'TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS',
    'TITAN.NS', 'ULTRACEMCO.NS'
]

# --- Functions ---
@st.cache_data
def fetch_stock_data(stocks, start_date, end_date):
    """Fetch stock data efficiently for all tickers."""
    try:
        data = yf.download(stocks, start=start_date, end=end_date, group_by="ticker")
        return data
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(data):
    """Calculate technical indicators."""
    indicators = {}
    for ticker in stocks:
        if ticker in data.columns:
            df = data[ticker]

            # Technical Indicators
            df['GK_Volatility'] = np.sqrt(0.5 * np.log(df['High'] / df['Low'])**2 - 
                                          (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open'])**2)

            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            sma = df['Close'].rolling(window=20).mean()
            std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = sma + (std * 2)
            df['BB_Middle'] = sma
            df['BB_Lower'] = sma - (std * 2)

            true_range = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            ], axis=1).max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()

            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            df['Rupee_Volume'] = df['Close'] * df['Volume']

            indicators[ticker] = df

    return indicators

def optimize_portfolio(indicators):
    """Use K-Means clustering for portfolio optimization."""
    features = []
    for ticker, df in indicators.items():
        if len(df) > 0:
            features.append([
                df['GK_Volatility'].mean(),
                df['RSI'].mean(),
                df['BB_Upper'].mean(),
                df['BB_Middle'].mean(),
                df['BB_Lower'].mean(),
                df['ATR'].mean(),
                df['MACD'].mean(),
                df['MACD_Signal'].mean(),
                df['Rupee_Volume'].mean()
            ])

    if len(features) == 0:
        st.error("No valid data to optimize portfolio.")
        return pd.DataFrame()

    features_df = pd.DataFrame(features, columns=[
        'GK_Volatility', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
        'ATR', 'MACD', 'MACD_Signal', 'Rupee_Volume'
    ], index=stocks[:len(features)])

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)

    kmeans = KMeans(n_clusters=3, random_state=0)
    features_df['Cluster'] = kmeans.fit_predict(scaled_features)

    return features_df

def recommend_stocks(features_df, top_n=5):
    """Recommend top stocks."""
    recommendations = []
    for cluster in features_df['Cluster'].unique():
        cluster_stocks = features_df[features_df['Cluster'] == cluster]
        ranked_stocks = cluster_stocks.sort_values(['GK_Volatility', 'RSI'], ascending=[True, False])
        recommendations.append(ranked_stocks.head(top_n))

    return pd.concat(recommendations)

# --- Streamlit UI ---
st.subheader("Sensex Stock Analysis")
start_date = st.date_input("Select Start Date", pd.to_datetime('2020-01-01'))
end_date = st.date_input("Select End Date", pd.to_datetime('today'))

if st.button("Analyze Stocks"):
    # Data processing
    data = fetch_stock_data(stocks, start_date, end_date)

    if not data.empty:
        indicators = calculate_technical_indicators(data)
        portfolio = optimize_portfolio(indicators)
        
        if not portfolio.empty:
            recommendations = recommend_stocks(portfolio)

            st.subheader("Recommended Stocks:")
            st.write(recommendations)

            # Explanation section
            st.markdown("""
            ### Explanation of Recommendations
            - **GK Volatility:** Measures stability. Lower values indicate more stable stocks.
            - **RSI:** Indicates if a stock is overbought (>70) or undervalued (<30).
            - **Bollinger Bands:** Show price volatility.
            - **ATR:** Measures market volatility. Lower ATR indicates stability.
            - **MACD:** Shows momentum trends.
            - **Rupee Volume:** Measures trading activity.
            """)
        else:
            st.warning("No valid portfolio optimization data.")
    else:
        st.warning("No stock data available for the selected date range.")
