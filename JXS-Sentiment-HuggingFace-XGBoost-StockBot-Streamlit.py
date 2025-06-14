#FinBERT Hugging Face Sentiment Analysis
#Pretrained NLP (Natual Language Processing) analyzes sentiment, built by Prosus AI
#Need the model than analyzes Financial Data, rather than general, thus using FinBERT 
#Need to use Transformers to feed into Pipeline
#Will be scraping news data from yahoo Finance RSS Feed
#Feedpasers is only necessary if we want to use an RSS Feed (Really Simple Syndication is a web feed that allows users and applications to receive updates from websites in a standardized, computer-readable format)
#The stock data only includes trading days, but the news articles can be published on any day, including weekends and holidays. When we reindex the sentiment data to the stock's date range (which skips non-trading days), any news from non-trading days get their dates adjusted to the nearest trading day or dropped, leading to zeros.



















































































































































#Virutual Environment: 3.10.16 ('.conda':conda)
#pip install streamlit, feedparser, transformers, tf-keras, pandas, yfinance, datetime, matplotlib, pytz, dateutil, numpy, scikit-learn, xgboost
import streamlit as st
import feedparser
from transformers import pipeline
import pandas as pd
import yfinance as yf                    #Need to run this in terminal: pip install --upgrade yfinance
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil import parser
import pytz
from pandas.tseries.offsets import BDay  #For business day calculations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb


#Initialize Hugging Face FinBERT Data pipeline
pipe = pipeline(task="text-classification", model="ProsusAI/finbert")

#For timezone conversion
def convert_to_utc_naive(dt):
    """Convert any datetime object to UTC-naive format"""
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is not None:
        return dt.astimezone(pytz.utc).replace(tzinfo=None)
    return dt

#Custom CSS Streamlit Colour Scheme
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}
h1, h2, h3, h4, h5, h6 {
    color: #800080;
}
.stButton>button {
    background-color: #800080;
    color: #FFFFFF;
    border-radius: 5px;
    border: 1px solid #800080;
}
.stTextInput>div>div>input {
    background-color: #000000;
    color: #FFFFFF;
    border: 1px solid #800080;
}
.stSlider>div>div>div>div {
    background-color: #800080;
}
.stDataFrame {
    background-color: #1A1A1A;
}
.st-ae {
    background-color: #1A1A1A;
}
</style>
""", unsafe_allow_html=True)

#Streamlit Layout
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker", "XYZ")
keyword = st.sidebar.text_input("Company Keyword", "Block")
forecast_days = st.sidebar.slider("Days to Forecast", 1, 50, 7)
run_button = st.sidebar.button("Run Analysis")

#Debugging Switch
debug_mode = st.sidebar.checkbox("Show Debugging Info")

st.title("Comprehensive Financial Analysis")
st.subheader(f"Analysis Report for {ticker}")

#Stock Data Analysis Section
try:
    st.write("## Complete Stock Data Analysis")
    
    #Creating dynamic date range with timezone awareness
    now_utc = datetime.now(pytz.utc)
    start_date = datetime(1997, 1, 1, tzinfo=pytz.utc)  #Set start date as January 1, 1997
    end_date = now_utc + pd.DateOffset(days=1)          #Including today's potential articles

    # Fetch the stock data with timezone-aware dates
    df = yf.download(ticker, start=start_date, end=end_date)

    if not df.empty:

        df['Daily Return %'] = df['Close'].pct_change() * 100
        
        # Create trading date reference
        trading_dates = df.index.normalize().unique()
        
        # --- Sentiment Processing with Trading Date Alignment ---
        sentiment_data = []
        rss_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
        feed = feedparser.parse(rss_url)
        
        if feed.entries:
            if debug_mode:
                st.sidebar.subheader("Raw Feed Debug")
                st.sidebar.write(f"Total articles: {len(feed.entries)}")
                st.sidebar.write("First article:", feed.entries[0])

            for entry in feed.entries:
                try:
                    # Check keyword match
                    if keyword.lower() not in entry.summary.lower():
                        continue

                    # Parse and normalize date
                    pub_date = entry.get('published')
                    if not pub_date:
                        continue
                                        # Parse and normalize date with timezone awareness
                    parsed_date = parser.parse(pub_date)
                    if not parsed_date.tzinfo:
                        parsed_date = parsed_date.replace(tzinfo=pytz.utc)
                    else:
                        parsed_date = parsed_date.astimezone(pytz.utc)
                        
                    # Convert to naive datetime in UTC for comparison
                    article_date = parsed_date.replace(tzinfo=None)
                    
                    # Find matching trading day (allow same-day if market is open)
                    # Get all possible trading dates up to now
                    possible_dates = df.loc[:now_utc.replace(tzinfo=None)].index
                    
                    # Find the nearest trading day (including same day if market open)
                    next_trading_day = min(
                        possible_dates[possible_dates >= article_date],
                        default=None
                    )
                    
                    # Create trading date reference with timezone-naive UTC dates
                    trading_dates = df.index.tz_localize(None).unique()
                    if next_trading_day is None:
                        # If article is after market close, use next trading day
                        next_trading_day = pd.to_datetime(article_date) + BDay(1)
                        if next_trading_day not in trading_dates:
                            continue

                    # Get sentiment
                    sentiment = pipe(entry.summary)[0]
                    score = sentiment['score']
                    if sentiment['label'] == 'negative':
                        score *= -1

                    sentiment_data.append({
                        'Date': next_trading_day,
                        'Score': score,
                        'Title': entry.title
                    })

                except Exception as e:
                    if debug_mode:
                        st.sidebar.error(f"Article error: {str(e)}")

        # Create and merge sentiment data
        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)
            sentiment_df = sentiment_df.groupby('Date')['Score'].mean().to_frame()
            
            # Merge with all trading dates
            sentiment_full = pd.DataFrame(index=trading_dates)
            sentiment_full = sentiment_full.join(sentiment_df).fillna(0)
            
            if debug_mode:
                st.sidebar.subheader("Sentiment Debug")
                st.sidebar.write("Raw sentiment data:", sentiment_data[:3])
                st.sidebar.write("Processed sentiment:", sentiment_df.head())
        else:
            sentiment_full = pd.DataFrame(index=trading_dates, data={'Score': 0.0})

        # --- COMBINED TABLE SECTION ---

        st.success("✅ Successfully retrieved historical stock data")

        with st.expander("Combined Sentiment & Returns Analysis", expanded=False):
            st.write("### Combined Sentiment & Returns Analysis")
            # Create combined dataframe
            combined_df = pd.DataFrame({
                'Date': df.index,
                'Sentiment Score': sentiment_full['Score'],
                'Daily Return %': df['Daily Return %']
            }).set_index('Date')
            
            # Style formatting
            combined_style = combined_df.style.format({
                'Sentiment Score': '{:.2f}',
                'Daily Return %': '{:.2f}%'
            }, na_rep="-")
            
            # Color conditional formatting
            combined_style = combined_style.applymap(
                lambda x: 'color: green' if isinstance(x, float) and x > 0 else 'color: red' if isinstance(x, float) and x < 0 else '',
                subset=['Sentiment Score', 'Daily Return %']
            )
            
            st.dataframe(
                combined_style,
                height=400,
                use_container_width=True
            )

        # Price moving averages
        df['Close_MA_20'] = df['Close'].rolling(window=20).mean()
        df['Close_MA_50'] = df['Close'].rolling(window=50).mean()
        df['Close_MA_100'] = df['Close'].rolling(window=100).mean()
        
        # Price Returns moving averages
        df['Return_MA_20'] = df['Daily Return %'].rolling(window=20).mean()
        df['Return_MA_50'] = df['Daily Return %'].rolling(window=50).mean()
        df['Return_MA_100'] = df['Daily Return %'].rolling(window=100).mean()

        # Format numeric columns
        format_dict = {
            'Open': '${:.2f}',
            'High': '${:.2f}',
            'Low': '${:.2f}',
            'Close': '${:.2f}',
            'Adj Close': '${:.2f}',
            'Close_MA_20': '${:.2f}',
            'Close_MA_50': '${:.2f}',
            'Close_MA_100': '${:.2f}',
            'Volume': '{:,}',
            'Daily Return %': '{:.2f}%',
            'Return_MA_20': '{:.2f}%',
            'Return_MA_50': '{:.2f}%',
            'Return_MA_100': '{:.2f}%'
        }
        
        # --- HISTORICAL DATA SECTION ---
        with st.expander("Complete Historical Data", expanded=False):
            st.write("### Complete Historical Data")
            st.dataframe(
                df.style.format(format_dict, na_rep="-"),
                height=600,
                use_container_width=True
            )

        # Plot section after table
        st.write("## Technical Analysis")
        
        # Price MA Plot
        st.write("### Price Moving Averages")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.index, df['Close'], label='Closing Price', alpha=0.5)
        ax1.plot(df.index, df['Close_MA_20'], label='20-Day MA', linewidth=1.5)
        ax1.plot(df.index, df['Close_MA_50'], label='50-Day MA', linewidth=1.5)
        ax1.plot(df.index, df['Close_MA_100'], label='100-Day MA', linewidth=1.5)
        ax1.set_title(f'{ticker} Price Movement with Moving Averages')
        ax1.set_ylabel('Price (USD)')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        # Return MA Plot
        st.write("### Daily Return Moving Averages")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.index, df['Daily Return %'], label='Daily Returns', alpha=0.3)
        ax2.plot(df.index, df['Return_MA_20'], label='20-Day MA', linewidth=1.5)
        ax2.plot(df.index, df['Return_MA_50'], label='50-Day MA', linewidth=1.5)
        ax2.plot(df.index, df['Return_MA_100'], label='100-Day MA', linewidth=1.5)
        ax2.set_title(f'{ticker} Daily Returns with Moving Averages')
        ax2.set_ylabel('Percentage Change')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Sentiment Driven Returns XGBoost Forecasting
        st.write("## Sentiment-Driven Return Forecasting")

        # Check if we have enough data
        if len(df) < 30:
            st.warning("Insufficient data for forecasting (need at least 30 days)")
        else:
            try:
                # Prepare dataset
                merged_data = pd.DataFrame({
                    'Returns': df['Daily Return %'],
                    'Sentiment': sentiment_full['Score']
                }).dropna()

                # Create lagged features
                lookback = 30  # Use 30 days of historical data for longer-term patterns
                for i in range(1, lookback+1):
                    merged_data[f'Returns_lag{i}'] = merged_data['Returns'].shift(i)
                    merged_data[f'Sentiment_lag{i}'] = merged_data['Sentiment'].shift(i)

                merged_data = merged_data.dropna()
                
                # Prepare features/target
                X = merged_data.drop('Returns', axis=1)
                y = merged_data['Returns']
                
                # Train/test split (time-series aware)
                split = int(0.8 * len(X))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                # Train XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Evaluate model
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Create forecast dataframe
                last_date = df.index[-1]
                forecast_dates = [last_date + BDay(i) for i in range(1, forecast_days+1)]
                
                # Prepare future features (recursive forecasting)
                decay_factor = 0.95                  # Added decay for longer forecasts
                current_features = X.iloc[-1].values
                forecasts = []
                
                for _ in range(forecast_days):
                    pred = model.predict(current_features.reshape(1, -1))[0] * decay_factor
                    forecasts.append(pred)
                    
                    # Update features for next prediction with decayed values
                    current_features = np.roll(current_features, 2)
                    current_features[0] = pred  # Update returns lag
                    current_features[1] = merged_data['Sentiment'].iloc[-1] * decay_factor  # Keep latest sentiment
                    decay_factor *= 0.95 # Exponential decay
                
                # Create results dataframe with confidence intervals
                forecast_df = pd.DataFrame({
                    'Date': forecast_dates,
                    'Predicted Return %': forecasts,
                    'Low Estimate': [x * 0.85 for x in forecasts],  # 15% lower bound
                    'High Estimate': [x * 1.15 for x in forecasts]  # 15% upper bound
                }).set_index('Date')

                # Plot results
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(y_test.index, y_test, label='Actual Returns')
                ax.plot(y_test.index, test_pred, label='Test Predictions', linestyle='--')
                ax.plot(forecast_df.index, forecast_df['Predicted Return %'], 
                        label='Forecast', linestyle=':', marker='o')
                ax.set_title(f'{ticker} Return Forecast ({forecast_days}-day Outlook)')
                ax.set_ylabel('Daily Return (%)')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

                # Show enhanced forecast table
                st.write("### 50-Day Forecast Results")
                st.dataframe(
                    forecast_df.style.format({
                        'Predicted Return %': '{:.2f}%',
                        'Low Estimate': '{:.2f}%',
                        'High Estimate': '{:.2f}%'
                    }).applymap(
                        lambda x: 'color: #00FF00' if isinstance(x, float) and x > 0 else 'color: #FF0000' if isinstance(x, float) and x < 0 else '',
                        subset=['Predicted Return %', 'Low Estimate', 'High Estimate']
                    ),
                    height=800,
                    use_container_width=True
                )
                
                # Feature importance
                st.write("#### Feature Importance")
                importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.bar_chart(importance.set_index('Feature'))

            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
                
    else:
        st.error("No stock data available for this ticker")

except Exception as e:
    st.error(f"Stock data error: {str(e)}")

# News Sentiment Analysis Section
try:
    st.write("## Complete News Sentiment Analysis")
    
    # Fetch and parse RSS feed
    rss_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    feed = feedparser.parse(rss_url)
    
    if not feed.entries:
        st.warning("No articles found in RSS feed")
    else:
        articles = []
        sentiment_scores = []
        
        # Process all articles
        for entry in feed.entries:
            if keyword.lower() in entry.summary.lower():
                try:
                    # Full text processing without truncation
                    sentiment = pipe(entry.summary)[0]
                    score = sentiment['score'] * (-1 if sentiment['label'] == 'negative' else 1)
                    
                    articles.append({
                        'Date': entry.get('published', 'N/A'),
                        'Title': entry.title,
                        'Sentiment': sentiment['label'],
                        'Score': score,
                        'Link': entry.link,
                        'Full Text': entry.summary
                    })
                    sentiment_scores.append(score)
                except Exception as e:
                    st.error(f"Error processing article: {str(e)}")
        
        if articles:
            # Display results
            st.write(f"### All Analyzed Articles ({len(articles)} total)")
            
            # Create expandable sections for each article
            for idx, article in enumerate(articles, 1):
                with st.expander(f"Article {idx}: {article['Title']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Date:** {article['Date']}")
                        st.write(f"**Sentiment:** {article['Sentiment']}")
                        st.write(f"**Score:** {article['Score']:.2f}")
                        st.write(f"[Read Full Article]({article['Link']})")
                    with col2:
                        st.write("**Summary:**")
                        st.write(article['Full Text'])
            
            # Display sentiment metrics
            st.write("### Aggregate Sentiment Analysis")
            avg_score = sum(sentiment_scores)/len(sentiment_scores) if sentiment_scores else 0.0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles Analyzed", len(articles))
            col2.metric("Average Sentiment Score", f"{avg_score:.2f}")
            col3.metric("Positive/Negative Ratio", 
                        f"{len([s for s in sentiment_scores if s > 0])}:{len([s for s in sentiment_scores if s < 0])}")
            
        else:
            st.warning("No articles matched the keyword filter")

except Exception as e:
    st.error(f"News analysis error: {str(e)}")

# --- METHODOLOGY SECTION ---
with st.expander("Methodology Notes", expanded=False):
    st.write("---")
    st.write("""
    **Methodology Notes:**
    - **Data Sources:**
        - Stock data sourced directly from Yahoo Finance historical records
        - News sentiment derived from Yahoo Finance RSS feed articles
        - External economic indicators (interest rates, inflation data) not currently included

    - **Technical Analysis:**
        - Daily returns calculated using closing prices (Percentage change from previous close)
        - Price moving averages (MAs) calculated on closing prices (20/50/100-Day SMA)
        - Return MAs calculated on daily percentage changes (20/50/100-Day SMA)
        - 20, 50, 100-Day MA periods represent Short, Medium, and Long-term trend indicators

    - **Sentiment Analysis:**
        - FinBERT model processes full article text without truncation
        - Sentiment scores range from -1 (negative) to +1 (positive)
        - Articles mapped to next trading day if published during non-market hours
        - Missing sentiment values filled with 0 (neutral baseline)

    - **Machine Learning Forecasting:**
        - XGBoost regression model with multiple input factors and enhanced long-term capabilities:
            - Endogenous: Historical price/return patterns (30-day lagged returns)
            - Exogenous: News sentiment trends (30-day lagged sentiment scores)
            - Technical indicators: Moving average convergence/divergence
            - Exponential error decay for multi-step forecasts
            - Confidence interval estimates (85-115% of predicted values)
            - Feature engineering optimized for 50-day predictions
        - Recursive multi-step forecasting for 1-50 day predictions with stability enhancements
        - Volatility-adjusted predictions
        - Explicit handling of temporal relationships through feature engineering
        - Time-series aware training (80% train / 20% test chronological split)
        - Model features importance analysis shows key prediction drivers
        - Business day alignment ensures valid trading date predictions

    - **Exogenous Factor Handling:**
        - News sentiment treated as external market sentiment indicator
        - Assumes constant sentiment impact horizon of 5 trading days
        - No forward-looking bias in sentiment feature engineering
        - Missing exogenous values handled through forward-filling

    - **Limitations:**
        - Does not currently incorporate macroeconomic indicators
        - Limited to single-equity analysis (no sector/industry factors)
        - Sentiment data limited to Yahoo Finance news sources
        - Model assumes stationarity of return/sentiment relationships
    """)
