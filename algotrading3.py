import streamlit as st
def page2(): 
    import streamlit as st
    st.title("Module 2")
    st.write("Welcome to Module 2!")    
    st.subheader("Understand Sensex and invest in it as a whole")
    import math
    import pandas as pd
    import yfinance as yf
    import numpy as np
    from scipy import stats
    from statistics import mean

    # Define the list of stock tickers
    # Replace this list with your actual tickers or load from a CSV
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

    # Function to split list into chunks
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # Function to get portfolio size from user
    def get_portfolio_size():
        """
        Prompts the user to enter the portfolio size and validates the input.
        
        Returns:
            float: The validated portfolio size.
        """
        while True:
            try:
                portfolio_size = float(input("Enter the value of your portfolio: "))
                if portfolio_size <= 0:
                    print("Portfolio size must be a positive number. Please try again.")
                    continue
                return portfolio_size
            except ValueError:
                print("That's not a valid number! Please enter a numerical value.")

    # =================== Momentum Strategy ===================

    # Function to build the initial Momentum DataFrame
    def build_initial_momentum_dataframe(stocks):
        """
        Fetches stock data using yfinance and builds the initial Momentum DataFrame.
        
        Parameters:
            stocks (list): List of stock ticker symbols.
            
        Returns:
            pd.DataFrame: Initial DataFrame with Ticker, Price, One-Year Price Return.
        """
        my_columns = ['Ticker', 'Price', 'One-Year Price Return', 'Number of Shares to Buy']
        momentum_data = []  # List to accumulate data

        # Split stocks into chunks of 100
        symbol_groups = list(chunks(stocks, 100))
        symbol_strings = [','.join(group) for group in symbol_groups]

        for symbol_string in symbol_strings:
            tickers = yf.Tickers(symbol_string)
            for symbol in symbol_string.split(','):
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info

                    # Fetch Price
                    price = info.get('regularMarketPrice', np.nan)
                    if np.isnan(price):
                        # Fallback to previousClose if regularMarketPrice is not available
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            price = hist['Close'].iloc[-1]
                        else:
                            price = np.nan

                    # Fetch One-Year Price Return
                    hist = ticker.history(period="1y")
                    if len(hist) < 1:
                        raise ValueError("Insufficient historical data.")
                    one_year_return = (hist['Close'][-1] / hist['Close'][0]) - 1

                    momentum_data.append({
                        'Ticker': symbol,
                        'Price': price,
                        'One-Year Price Return': one_year_return,
                        'Number of Shares to Buy': 'N/A'  # Placeholder
                    })
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    momentum_data.append({
                        'Ticker': symbol,
                        'Price': 'N/A',
                        'One-Year Price Return': 'N/A',
                        'Number of Shares to Buy': 'N/A'
                    })

        momentum_df = pd.DataFrame(momentum_data, columns=my_columns)
        return momentum_df

    # Function to filter top 50 momentum stocks
    def filter_top_momentum_stocks(momentum_df, top_n=50):
        """
        Sorts the DataFrame by One-Year Price Return and selects top N momentum stocks.
        
        Parameters:
            momentum_df (pd.DataFrame): Initial Momentum DataFrame with stock data.
            top_n (int): Number of top momentum stocks to select.
            
        Returns:
            pd.DataFrame: Filtered DataFrame with top N momentum stocks.
        """
        # Remove rows with 'N/A' in One-Year Price Return
        filtered_df = momentum_df[momentum_df['One-Year Price Return'] != 'N/A'].copy()

        # Sort by One-Year Price Return descending
        filtered_df.sort_values('One-Year Price Return', ascending=False, inplace=True)

        # Select top_n stocks
        top_momentum_df = filtered_df.head(top_n).reset_index(drop=True)
        return top_momentum_df

    # Function to calculate number of shares to buy for Momentum Strategy
    def calculate_shares_momentum(final_df, portfolio_size):
        """
        Calculates the number of shares to buy for each stock in the Momentum portfolio.
        
        Parameters:
            final_df (pd.DataFrame): Final Momentum DataFrame with selected stocks.
            portfolio_size (float): Total portfolio value.
            
        Returns:
            pd.DataFrame: Final Momentum DataFrame with calculated shares to buy.
        """
        position_size = portfolio_size / len(final_df.index)
        
        for i in final_df.index:
            try:
                price = final_df.at[i, 'Price']
                if isinstance(price, (int, float)) and price > 0:
                    final_df.at[i, 'Number of Shares to Buy'] = math.floor(position_size / price)
                else:
                    final_df.at[i, 'Number of Shares to Buy'] = 'N/A'
            except Exception as e:
                print(f"Error calculating shares for {final_df.at[i, 'Ticker']}: {e}")
                final_df.at[i, 'Number of Shares to Buy'] = 'N/A'
        
        return final_df

    # =================== Value Strategy ===================

    # Function to build the initial Value DataFrame
    def build_initial_value_dataframe(stocks):
        """
        Fetches stock data using yfinance and builds the initial Value DataFrame.
        
        Parameters:
            stocks (list): List of stock ticker symbols.
            
        Returns:
            pd.DataFrame: Initial DataFrame with Ticker, Price, Price-to-Earnings Ratio.
        """
        my_columns = ['Ticker', 'Price', 'Price-to-Earnings Ratio', 'Number of Shares to Buy']
        value_data = []  # List to accumulate data

        # Split stocks into chunks of 100
        symbol_groups = list(chunks(stocks, 100))
        symbol_strings = [','.join(group) for group in symbol_groups]

        for symbol_string in symbol_strings:
            tickers = yf.Tickers(symbol_string)
            for symbol in symbol_string.split(','):
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info

                    # Fetch Price
                    price = info.get('regularMarketPrice', np.nan)
                    if np.isnan(price):
                        # Fallback to previousClose if regularMarketPrice is not available
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            price = hist['Close'].iloc[-1]
                        else:
                            price = np.nan

                    # Fetch Price-to-Earnings Ratio (trailingPE)
                    pe_ratio = info.get('trailingPE', np.nan)

                    value_data.append({
                        'Ticker': symbol,
                        'Price': price,
                        'Price-to-Earnings Ratio': pe_ratio,
                        'Number of Shares to Buy': 'N/A'  # Placeholder
                    })
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    value_data.append({
                        'Ticker': symbol,
                        'Price': 'N/A',
                        'Price-to-Earnings Ratio': 'N/A',
                        'Number of Shares to Buy': 'N/A'
                    })

        value_df = pd.DataFrame(value_data, columns=my_columns)
        return value_df

    # Function to filter top 50 value stocks based on PE Ratio
    def filter_top_value_stocks(value_df, top_n=50):
        """
        Sorts the DataFrame by Price-to-Earnings Ratio and selects top N value stocks.
        
        Parameters:
            value_df (pd.DataFrame): Initial Value DataFrame with stock data.
            top_n (int): Number of top value stocks to select.
            
        Returns:
            pd.DataFrame: Filtered DataFrame with top N value stocks.
        """
        # Remove rows with 'N/A' in Price-to-Earnings Ratio
        filtered_df = value_df[value_df['Price-to-Earnings Ratio'] != 'N/A'].copy()

        # Remove rows with non-positive PE Ratio
        filtered_df = filtered_df[filtered_df['Price-to-Earnings Ratio'] > 0]

        # Sort by Price-to-Earnings Ratio ascending (lower is better for value stocks)
        filtered_df.sort_values('Price-to-Earnings Ratio', ascending=True, inplace=True)

        # Select top_n stocks
        top_value_df = filtered_df.head(top_n).reset_index(drop=True)
        return top_value_df

    # Function to build Robust Value (RV) DataFrame with additional metrics
    def build_rv_dataframe(stocks):
        """
        Fetches additional stock data and builds the Robust Value DataFrame.
        
        Parameters:
            stocks (list): List of stock ticker symbols.
            
        Returns:
            pd.DataFrame: RV DataFrame with multiple valuation metrics.
        """
        rv_columns = [
            'Ticker',
            'Price',
            'Number of Shares to Buy', 
            'Price-to-Earnings Ratio',
            'PE Percentile',
            'Price-to-Book Ratio',
            'PB Percentile',
            'Price-to-Sales Ratio',
            'PS Percentile',
            'EV/EBITDA',
            'EV/EBITDA Percentile',
            'EV/GP',
            'EV/GP Percentile',
            'RV Score'
        ]
        rv_data = []
        
        # Split stocks into chunks of 100
        symbol_groups = list(chunks(stocks, 100))
        symbol_strings = [','.join(group) for group in symbol_groups]

        for symbol_string in symbol_strings:
            tickers = yf.Tickers(symbol_string)
            for symbol in symbol_string.split(','):
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info

                    # Fetch Price
                    price = info.get('regularMarketPrice', np.nan)
                    if np.isnan(price):
                        # Fallback to previousClose if regularMarketPrice is not available
                        hist = ticker.history(period="1d")
                        if not hist.empty:
                            price = hist['Close'].iloc[-1]
                        else:
                            price = np.nan

                    # Fetch Valuation Metrics
                    pe_ratio = info.get('trailingPE', np.nan)
                    pb_ratio = info.get('priceToBook', np.nan)
                    ps_ratio = info.get('priceToSalesTrailing12Months', np.nan)
                    enterprise_value = info.get('enterpriseValue', np.nan)
                    ebitda = info.get('ebitda', np.nan)
                    gross_profit = info.get('grossProfits', np.nan)

                    # Calculate EV/EBITDA and EV/GP
                    ev_to_ebitda = enterprise_value / ebitda if ebitda and ebitda != 0 else np.nan
                    ev_to_gross_profit = enterprise_value / gross_profit if gross_profit and gross_profit != 0 else np.nan

                    rv_data.append({
                        'Ticker': symbol,
                        'Price': price,
                        'Number of Shares to Buy': 'N/A',  # Placeholder
                        'Price-to-Earnings Ratio': pe_ratio,
                        'PE Percentile': 'N/A',
                        'Price-to-Book Ratio': pb_ratio,
                        'PB Percentile': 'N/A',
                        'Price-to-Sales Ratio': ps_ratio,
                        'PS Percentile': 'N/A',
                        'EV/EBITDA': ev_to_ebitda,
                        'EV/EBITDA Percentile': 'N/A',
                        'EV/GP': ev_to_gross_profit,
                        'EV/GP Percentile': 'N/A',
                        'RV Score': 'N/A'
                    })
                except Exception as e:
                    print(f"Error fetching data for {symbol}: {e}")
                    rv_data.append({
                        'Ticker': symbol,
                        'Price': 'N/A',
                        'Number of Shares to Buy': 'N/A',
                        'Price-to-Earnings Ratio': 'N/A',
                        'PE Percentile': 'N/A',
                        'Price-to-Book Ratio': 'N/A',
                        'PB Percentile': 'N/A',
                        'Price-to-Sales Ratio': 'N/A',
                        'PS Percentile': 'N/A',
                        'EV/EBITDA': 'N/A',
                        'EV/EBITDA Percentile': 'N/A',
                        'EV/GP': 'N/A',
                        'EV/GP Percentile': 'N/A',
                        'RV Score': 'N/A'
                    })

        rv_df = pd.DataFrame(rv_data, columns=rv_columns)
        return rv_df

    # Function to handle missing data by filling with column means
    def handle_missing_data(rv_df):
        """
        Replaces missing data in specific columns with the column mean.
        
        Parameters:
            rv_df (pd.DataFrame): RV DataFrame with potential missing data.
            
        Returns:
            pd.DataFrame: RV DataFrame with missing data handled.
        """
        columns_to_fill = ['Price-to-Earnings Ratio', 'Price-to-Book Ratio', 
                        'Price-to-Sales Ratio', 'EV/EBITDA', 'EV/GP']
        
        for column in columns_to_fill:
            # Convert to numeric, coerce errors to NaN
            rv_df[column] = pd.to_numeric(rv_df[column], errors='coerce')
            
            if rv_df[column].isnull().any():
                mean_value = rv_df[column].mean()
                rv_df[column].fillna(mean_value, inplace=True)
        
        return rv_df

    # Function to calculate percentiles for valuation metrics
    def calculate_value_percentiles(rv_df):
        """
        Calculates percentile scores for different valuation metrics.
        
        Parameters:
            rv_df (pd.DataFrame): RV DataFrame with valuation metrics.
            
        Returns:
            pd.DataFrame: RV DataFrame with calculated percentiles.
        """
        metrics = {
            'Price-to-Earnings Ratio': 'PE Percentile',
            'Price-to-Book Ratio': 'PB Percentile',
            'Price-to-Sales Ratio': 'PS Percentile',
            'EV/EBITDA': 'EV/EBITDA Percentile',
            'EV/GP': 'EV/GP Percentile'
        }
        
        for metric, percentile_col in metrics.items():
            # Ensure the metric column is numeric
            rv_df[metric] = pd.to_numeric(rv_df[metric], errors='coerce')
            
            # Calculate percentile ranks
            rv_df[percentile_col] = rv_df[metric].rank(pct=True)
        
        return rv_df

    # Function to calculate RV Score
    def calculate_rv_score(rv_df):
        """
        Calculates the RV Score as the mean of the valuation percentiles.
        
        Parameters:
            rv_df (pd.DataFrame): RV DataFrame with percentile scores.
            
        Returns:
            pd.DataFrame: RV DataFrame with RV Score.
        """
        metrics_percentiles = [
            'PE Percentile',
            'PB Percentile',
            'PS Percentile',
            'EV/EBITDA Percentile',
            'EV/GP Percentile'
        ]
        
        rv_df['RV Score'] = rv_df[metrics_percentiles].mean(axis=1)
        return rv_df

    # Function to select top 50 value stocks based on RV Score
    def select_top_rv_stocks(rv_df, top_n=50):
        """
        Sorts the DataFrame by RV Score and selects top N value stocks.
        
        Parameters:
            rv_df (pd.DataFrame): RV DataFrame with RV Scores.
            top_n (int): Number of top value stocks to select.
            
        Returns:
            pd.DataFrame: Filtered RV DataFrame with top N value stocks.
        """
        # Remove rows with 'N/A' in RV Score
        filtered_df = rv_df[rv_df['RV Score'].notna()].copy()

        # Sort by RV Score descending (higher is better)
        filtered_df.sort_values(by='RV Score', ascending=False, inplace=True)

        # Select top_n stocks
        top_rv_df = filtered_df.head(top_n).reset_index(drop=True)
        return top_rv_df

    # Function to calculate number of shares to buy for Value Strategy
    def calculate_shares_value(final_df, portfolio_size):
        """
        Calculates the number of shares to buy for each stock in the Value portfolio.
        
        Parameters:
            final_df (pd.DataFrame): Final Value DataFrame with selected stocks.
            portfolio_size (float): Total portfolio value.
            
        Returns:
            pd.DataFrame: Final Value DataFrame with calculated shares to buy.
        """
        position_size = portfolio_size / len(final_df.index)
        
        for i in final_df.index:
            try:
                price = final_df.at[i, 'Price']
                if isinstance(price, (int, float)) and price > 0:
                    final_df.at[i, 'Number of Shares to Buy'] = math.floor(position_size / price)
                else:
                    final_df.at[i, 'Number of Shares to Buy'] = 'N/A'
            except Exception as e:
                print(f"Error calculating shares for {final_df.at[i, 'Ticker']}: {e}")
                final_df.at[i, 'Number of Shares to Buy'] = 'N/A'
        
        return final_df

    # =================== Combined Strategy ===================

    # Function to build the complete Momentum strategy
    def build_momentum_strategy(stocks):
        """
        Builds the complete Momentum strategy DataFrame.
        
        Parameters:
            stocks (list): List of stock ticker symbols.
            
        Returns:
            pd.DataFrame: Final Momentum DataFrame ready for portfolio allocation.
        """
        # Step 1: Build initial momentum DataFrame
        initial_momentum_df = build_initial_momentum_dataframe(stocks)
        
        # Step 2: Filter top 50 momentum stocks
        top_momentum_df = filter_top_momentum_stocks(initial_momentum_df, top_n=50)
        
        return top_momentum_df

    # Function to build the complete Value strategy
    def build_value_strategy(stocks):
        """
        Builds the complete Value strategy DataFrame.
        
        Parameters:
            stocks (list): List of stock ticker symbols.
            
        Returns:
            pd.DataFrame: Final Value DataFrame ready for portfolio allocation.
        """
        # Step 1: Build initial value DataFrame
        initial_value_df = build_initial_value_dataframe(stocks)
        
        # Step 2: Filter top 50 value stocks based on PE Ratio
        top_value_df = filter_top_value_stocks(initial_value_df, top_n=50)
        
        # Step 3: Build RV DataFrame with additional valuation metrics
        rv_df = build_rv_dataframe(top_value_df['Ticker'].tolist())
        
        # Step 4: Handle missing data
        rv_df = handle_missing_data(rv_df)
        
        # Step 5: Calculate percentiles
        rv_df = calculate_value_percentiles(rv_df)
        
        # Step 6: Calculate RV Score
        rv_df = calculate_rv_score(rv_df)
        
        # Step 7: Select top 50 RV stocks
        final_rv_df = select_top_rv_stocks(rv_df, top_n=50)
        
        return final_rv_df

    # Main execution flow

    import streamlit as st
    import pandas as pd
    import yfinance as yf
    import numpy as np
    import math
    from scipy import stats
    from statistics import mean

    # Function definitions (keep all the functions from the original script here)
    # ...

    # Streamlit app
    
    

    st.subheader("Full Sensex Portfolio Strategy Builder")

    st.sidebar.header("Configuration")
    portfolio_size = st.sidebar.number_input("Enter the value of your portfolio in rupees:", min_value=1000, value=100000, step=1000)

    # Use a smaller subset of stocks for demonstration purposes
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

    if st.sidebar.button("Generate Portfolios"):
        with st.spinner("Building Momentum Strategy..."):
            final_momentum_df = build_momentum_strategy(stocks)
            final_momentum_df = calculate_shares_momentum(final_momentum_df, portfolio_size)

        with st.spinner("Building Value Strategy..."):
            final_value_df = build_value_strategy(stocks)
            final_value_df = calculate_shares_value(final_value_df, portfolio_size)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Momentum Portfolio")
            st.dataframe(final_momentum_df)
            
            csv_momentum = final_momentum_df.to_csv(index=False)
            st.download_button(
                label="Download Momentum Portfolio CSV",
                data=csv_momentum,
                file_name="momentum_portfolio.csv",
                mime="text/csv",
            )

        with col2:
            st.subheader("Value Portfolio")
            st.dataframe(final_value_df)
            
            csv_value = final_value_df.to_csv(index=False)
            st.download_button(
                label="Download Value Portfolio CSV",
                data=csv_value,
                file_name="value_portfolio.csv",
                mime="text/csv",
            )

    st.sidebar.markdown("---")
    st.sidebar.info("""
    This module builds quant-based and AI-powered strategies based on Sensex stocks as of 2024.
    
    1. Enter your portfolio value in the sidebar.
    2. Click 'Generate Portfolios' to create Momentum and Value portfolios.
    3. View the results and download CSV files for further analysis.
    4. For the amount you entered, you will get the number of shares of a stock to buy under momentum and value section.
    5. Momentum is dependent on how a stock trends over time. Value is dependent on a company's fundamentals.
    6. Robust Value ( RV) is a powerful indicator of a stock.

    Note: 1.If RV score in value strategy is more, buy the stock.
        2.If one-year price return in momentum strategy is more, buy the stock.       
    """)

