import streamlit as st 
def page3():
    import streamlit as st
    st.title("Module 3")
    st.write(" Welcome to Module 3")
    st.subheader("Understand Sensex and invest in it as a whole")

    import streamlit as st
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

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

    def fetch_stock_data(stocks, start_date, end_date):
        """Fetch stock data for the given tickers and date range using Ticker objects."""
        data = pd.DataFrame()
        for ticker in stocks:
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            for column in hist.columns:
                data[(column, ticker)] = hist[column]
        return data

    def calculate_technical_indicators(data):
        """Calculate various technical indicators."""
        for ticker in stocks:
            close = data[('Close', ticker)]
            high = data[('High', ticker)]
            low = data[('Low', ticker)]
            open_ = data[('Open', ticker)]
            volume = data[('Volume', ticker)]

            # Garman-Klass Volatility
            data[(f'GK_Volatility', ticker)] = np.sqrt(
                0.5 * np.log(high / low)**2 - (2 * np.log(2) - 1) * np.log(close / open_)**2
            )
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data[(f'RSI', ticker)] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma = close.rolling(window=20).mean()
            std = close.rolling(window=20).std()
            data[(f'BB_Upper', ticker)] = sma + (std * 2)
            data[(f'BB_Middle', ticker)] = sma
            data[(f'BB_Lower', ticker)] = sma - (std * 2)
            
            # ATR
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            data[(f'ATR', ticker)] = true_range.rolling(14).mean()
            
            # MACD
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            data[(f'MACD', ticker)] = exp1 - exp2
            data[(f'MACD_Signal', ticker)] = data[(f'MACD', ticker)].ewm(span=9, adjust=False).mean()
            
            # Rupee Volume
            data[(f'Rupee_Volume', ticker)] = close * volume

        return data

    def optimize_portfolio(data):
        """Use K-Means clustering for portfolio optimization based on technical indicators."""
        # Prepare the feature set for clustering
        features = []
        for ticker in stocks:
            features.append([
                data[(f'GK_Volatility', ticker)].mean(),
                data[(f'RSI', ticker)].mean(),
                data[(f'BB_Upper', ticker)].mean(),
                data[(f'BB_Middle', ticker)].mean(),
                data[(f'BB_Lower', ticker)].mean(),
                data[(f'ATR', ticker)].mean(),
                data[(f'MACD', ticker)].mean(),
                data[(f'MACD_Signal', ticker)].mean(),
                data[(f'Rupee_Volume', ticker)].mean()
            ])
        
        # Create a DataFrame for clustering
        features_df = pd.DataFrame(features, columns=[
            'GK_Volatility', 'RSI', 'BB_Upper', 'BB_Middle', 'BB_Lower',
            'ATR', 'MACD', 'MACD_Signal', 'Rupee_Volume'
        ], index=stocks)
        
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features_df)

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(scaled_features)
        
        # Assign cluster labels to each stock
        features_df['Cluster'] = kmeans.labels_
        
        return features_df

    def recommend_stocks(features_df, top_n=5):
        """Recommend multiple stocks based on clustering and technical indicators."""
        recommendations = []
        
        # Select the top N stocks from each cluster based on combined criteria
        for cluster in features_df['Cluster'].unique():
            cluster_stocks = features_df[features_df['Cluster'] == cluster]
            
            # Rank by a combination of indicators (e.g., low volatility, high RSI)
            ranked_stocks = cluster_stocks.sort_values(by=['GK_Volatility', 'RSI'], ascending=[True, False])
            
            # Select top N stocks from this cluster
            top_stocks = ranked_stocks.head(top_n)
            recommendations.append(top_stocks)

        # Combine all recommendations into one DataFrame
        recommendations_df = pd.concat(recommendations)
        
        return recommendations_df

    # Streamlit UI
    st.subheader("Sensex Stock Analysis and Recommendations")
    st.write("Know the historical performance of Sensex Stocks")

    # User input for date range
    start_date = st.date_input("Select Start Date", value=pd.to_datetime('2010-01-01'))
    end_date = st.date_input("Select End Date", value=pd.to_datetime('today'))

    if st.button("Analyze Stocks"):
        # Fetch data
        data = fetch_stock_data(stocks, start_date, end_date)
        
        # Calculate technical indicators
        data = calculate_technical_indicators(data)
        
        # Optimize portfolio using K-Means clustering
        portfolio_optimization_results = optimize_portfolio(data)
        
        # Recommend stocks based on clustering results
        recommended_stocks = recommend_stocks(portfolio_optimization_results)
        
        # Display results
        st.subheader("Recommended Stocks to Buy:")
        st.write(recommended_stocks)

        st.markdown("""
            ### Explanation of Recommendations
            - **GK_Volatility**: Lower values indicate less price fluctuation, making the stock more stable.
            - **RSI**: The Relative Strength Index ranges from 0 to 100. A value below 30 suggests the stock is undervalued, while a value above 70 indicates it might be overvalued.
            - **Bollinger Bands (BB)**: These bands indicate volatility. When prices are near the upper band, the stock may be overbought; near the lower band, it may be oversold.
            - **ATR**: Average True Range measures market volatility. Lower ATR values suggest a more stable market.
            - **MACD**: Moving Average Convergence Divergence indicates momentum. A positive value suggests an upward trend.
            - **Rupee Volume**: This indicates the total market activity in terms of money. Higher values suggest more interest in the stock.
            
            Stocks with lower volatility and favorable indicators (like high RSI) are generally recommended for purchase as they represent potential growth opportunities while minimizing risks.
            """)

