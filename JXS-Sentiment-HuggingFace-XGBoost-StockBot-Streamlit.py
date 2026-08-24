# FinBERT Hugging Face Sentiment Analysis + XGBoost Streamlit StockBot
# Updated to handle yfinance rate limits/empty downloads, yfinance MultiIndex columns,
# Streamlit/PyTorch watcher issues, and safer conclusion rendering.

import os

# Helps prevent Streamlit Cloud from crashing while watching PyTorch internals.
# For deployment, also add .streamlit/config.toml with fileWatcherType = "none".
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st
import feedparser
from transformers import pipeline
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil import parser
import pytz
from pandas.tseries.offsets import BDay
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Extra workaround for Streamlit + torch.classes watcher bug.
try:
    import torch
    torch.classes.__path__ = []
except Exception:
    pass

st.set_page_config(page_title="JXS StockBot", layout="wide")


# -----------------------------
# Cached resources / data
# -----------------------------
@st.cache_resource(show_spinner="Loading FinBERT model...")
def load_finbert_pipeline():
    return pipeline(task="text-classification", model="ProsusAI/finbert")


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_stock_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """Download stock data and normalize yfinance output."""
    raw_df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return normalize_yfinance_frame(raw_df, ticker)


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_yahoo_rss(ticker: str):
    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    return feedparser.parse(rss_url)


# -----------------------------
# Helper functions
# -----------------------------
def normalize_yfinance_frame(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Flatten yfinance MultiIndex columns so df['Close'] works reliably."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        ticker_upper = ticker.upper()
        level_0 = [str(x).upper() for x in df.columns.get_level_values(0)]
        level_1 = [str(x).upper() for x in df.columns.get_level_values(1)]

        if ticker_upper in level_0:
            original_value = df.columns.get_level_values(0)[level_0.index(ticker_upper)]
            df = df.xs(original_value, axis=1, level=0, drop_level=True)
        elif ticker_upper in level_1:
            original_value = df.columns.get_level_values(1)[level_1.index(ticker_upper)]
            df = df.xs(original_value, axis=1, level=1, drop_level=True)
        elif len(df.columns.get_level_values(0).unique()) == 1:
            df.columns = df.columns.droplevel(0)
        elif len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = df.columns.droplevel(1)
        else:
            df.columns = ["_".join(str(part) for part in col if str(part)) for col in df.columns]

    df.columns = [str(col).strip() for col in df.columns]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def get_entry_text(entry) -> str:
    return (entry.get("summary") or entry.get("title") or "").strip()


def get_entry_title(entry) -> str:
    return (entry.get("title") or "Untitled").strip()


def sentiment_score(text: str, pipe) -> tuple[str, float]:
    if not text:
        return "neutral", 0.0
    # FinBERT/BERT models have token limits, so use truncation to avoid runtime errors.
    result = pipe(text, truncation=True)[0]
    label = result.get("label", "neutral")
    score = float(result.get("score", 0.0))
    if label.lower() == "negative":
        score *= -1
    return label, score


def align_to_trading_day(pub_date: str, trading_dates: pd.DatetimeIndex):
    """Map an article publish date to the same or next available trading day."""
    if not pub_date:
        return None

    parsed_date = parser.parse(pub_date)
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=pytz.utc)
    else:
        parsed_date = parsed_date.astimezone(pytz.utc)

    article_day = pd.Timestamp(parsed_date.date())
    matching_dates = trading_dates[trading_dates >= article_day]
    if len(matching_dates) == 0:
        return None
    return matching_dates[0]


def color_value(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        if x > 0:
            return "color: green"
        if x < 0:
            return "color: red"
    return ""


def style_with_colors(df: pd.DataFrame, format_dict: dict, subset=None):
    styled = df.style.format(format_dict, na_rep="-")
    try:
        return styled.map(color_value, subset=subset)
    except AttributeError:
        # Older pandas fallback.
        return styled.applymap(color_value, subset=subset)


def make_sentiment_dataframe(feed, keyword: str, trading_dates: pd.DatetimeIndex, pipe, debug_mode=False):
    sentiment_data = []
    keyword_clean = keyword.strip().lower()

    if not getattr(feed, "entries", None):
        return pd.DataFrame(index=trading_dates, data={"Score": 0.0}), []

    for entry in feed.entries:
        try:
            text = get_entry_text(entry)
            if keyword_clean and keyword_clean not in text.lower() and keyword_clean not in get_entry_title(entry).lower():
                continue

            next_trading_day = align_to_trading_day(entry.get("published"), trading_dates)
            if next_trading_day is None:
                continue

            label, score = sentiment_score(text, pipe)
            sentiment_data.append(
                {
                    "Date": next_trading_day,
                    "Score": score,
                    "Sentiment": label,
                    "Title": get_entry_title(entry),
                    "Link": entry.get("link", ""),
                    "Full Text": text,
                    "Published": entry.get("published", "N/A"),
                }
            )
        except Exception as e:
            if debug_mode:
                st.sidebar.error(f"Article error: {e}")

    if sentiment_data:
        sentiment_df = (
            pd.DataFrame(sentiment_data)
            .groupby("Date", as_index=True)["Score"]
            .mean()
            .to_frame()
        )
        sentiment_full = pd.DataFrame(index=trading_dates)
        sentiment_full = sentiment_full.join(sentiment_df).fillna(0.0)
    else:
        sentiment_full = pd.DataFrame(index=trading_dates, data={"Score": 0.0})

    return sentiment_full, sentiment_data


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Daily Return %"] = df["Close"].pct_change() * 100
    df["Close_MA_20"] = df["Close"].rolling(window=20).mean()
    df["Close_MA_50"] = df["Close"].rolling(window=50).mean()
    df["Close_MA_100"] = df["Close"].rolling(window=100).mean()
    df["Return_MA_20"] = df["Daily Return %"].rolling(window=20).mean()
    df["Return_MA_50"] = df["Daily Return %"].rolling(window=50).mean()
    df["Return_MA_100"] = df["Daily Return %"].rolling(window=100).mean()
    return df


def build_forecast(df: pd.DataFrame, sentiment_full: pd.DataFrame, forecast_days: int, ticker: str):
    lookback = 30
    merged_data = pd.DataFrame(
        {
            "Returns": df["Daily Return %"],
            "Sentiment": sentiment_full["Score"].reindex(df.index).fillna(0.0),
        }
    ).dropna()

    for i in range(1, lookback + 1):
        merged_data[f"Returns_lag{i}"] = merged_data["Returns"].shift(i)
        merged_data[f"Sentiment_lag{i}"] = merged_data["Sentiment"].shift(i)

    merged_data = merged_data.dropna()
    if len(merged_data) < 50:
        st.warning("Insufficient usable data for forecasting after lag creation. Try a ticker with more trading history.")
        return

    X = merged_data.drop("Returns", axis=1)
    y = merged_data["Returns"]

    split = int(0.8 * len(X))
    if split <= 0 or split >= len(X):
        st.warning("Insufficient train/test data for forecasting.")
        return

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    test_pred = model.predict(X_test)

    last_date = df.index[-1]
    forecast_dates = [last_date + BDay(i) for i in range(1, forecast_days + 1)]

    history_returns = merged_data["Returns"].iloc[-lookback:].tolist()
    history_sentiment = merged_data["Sentiment"].iloc[-lookback:].tolist()
    latest_sentiment = float(history_sentiment[-1]) if history_sentiment else 0.0
    decay_factor = 0.95
    forecasts = []

    for _ in range(forecast_days):
        row = {"Sentiment": latest_sentiment * decay_factor}
        for i in range(1, lookback + 1):
            row[f"Returns_lag{i}"] = history_returns[-i]
            row[f"Sentiment_lag{i}"] = history_sentiment[-i]

        row_df = pd.DataFrame([row], columns=X.columns)
        pred = float(model.predict(row_df)[0]) * decay_factor
        forecasts.append(pred)

        history_returns.append(pred)
        history_sentiment.append(row["Sentiment"])
        decay_factor *= 0.95

    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Predicted Return %": forecasts,
            "Low Estimate": [x * 0.85 for x in forecasts],
            "High Estimate": [x * 1.15 for x in forecasts],
        }
    ).set_index("Date")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label="Actual Returns")
    ax.plot(y_test.index, test_pred, label="Test Predictions", linestyle="--")
    ax.plot(forecast_df.index, forecast_df["Predicted Return %"], label="Forecast", linestyle=":", marker="o")
    ax.set_title(f"{ticker} Return Forecast ({forecast_days}-day Outlook)")
    ax.set_ylabel("Daily Return (%)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.write(f"### {forecast_days}-Day Forecast Results")
    st.dataframe(
        style_with_colors(
            forecast_df,
            {
                "Predicted Return %": "{:.2f}%",
                "Low Estimate": "{:.2f}%",
                "High Estimate": "{:.2f}%",
            },
            subset=["Predicted Return %", "Low Estimate", "High Estimate"],
        ),
        height=400,
        use_container_width=True,
    )

    st.write("### Model Performance Metrics")
    mae = mean_absolute_error(y_test, test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, test_pred))

    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}%")
    with metrics_col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}%")

    st.write("#### Feature Importance")
    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    st.bar_chart(importance.set_index("Feature"))


# -----------------------------
# CSS / Layout
# -----------------------------
st.markdown(
    """
<style>
.stApp { background-color: #000000; color: #FFFFFF; }
h1, h2, h3, h4, h5, h6 { color: #800080; }
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
.stSlider>div>div>div>div { background-color: #800080; }
.stDataFrame { background-color: #1A1A1A; }
.st-ae { background-color: #1A1A1A; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("IRIS Stock Financial Sentiment Console")

with st.sidebar.form("analysis_form"):
    st.header("Input Parameters")
    ticker = st.text_input("Stock Ticker", "^SPX").strip().upper()
    keyword = st.text_input("Company Keyword", "S&P 500").strip()
    forecast_days = st.slider("Days to Forecast", 1, 50, 7)
    debug_mode = st.checkbox("Show Debugging Info")
    run_button = st.form_submit_button("Run IRIS Scan")

if not run_button:
    st.info(
        """**IRIS**, named after the Greek messenger goddess and the colors of the rainbow, is a stock analysis dashboard that delivers real-time news sentiment insights through Hugging Face's FinBERT NLP, combining market data, technical indicators, and XGBoost forecasting to help users interpret potential stock trends.

Enter a stock ticker in the sidebar at the top-left corner. Click **Run IRIS Scan** to start."""
    )
    st.stop()

if not ticker:
    st.error("Please enter a valid stock ticker.")
    st.stop()

st.subheader(f"Analysis Report for {ticker}")

pipe = load_finbert_pipeline()

# -----------------------------
# Stock Data Analysis Section
# -----------------------------
stock_data_available = False
sentiment_full = pd.DataFrame()
articles_from_stock_section = []

st.write("## Complete Stock Data Analysis")
now_utc = datetime.now(pytz.utc)
start_date = datetime(1997, 1, 1)
end_date = (now_utc + pd.DateOffset(days=1)).to_pydatetime().replace(tzinfo=None)

try:
    df = fetch_stock_data(ticker, start_date, end_date)
except Exception as e:
    df = pd.DataFrame()
    st.error(f"Stock data download failed: {e}")

if df.empty:
    st.warning(
        "No stock data was returned. This can happen if Yahoo Finance is rate-limiting the app or if the ticker is invalid. "
        "Try again later, try a different ticker, or clear the Streamlit cache/redeploy."
    )
else:
    required_columns = {"Close"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        st.error(f"Stock data was returned, but required columns are missing: {', '.join(sorted(missing_columns))}")
        if debug_mode:
            st.sidebar.write("Returned columns:", list(df.columns))
    else:
        stock_data_available = True
        df = add_technical_features(df)
        trading_dates = pd.DatetimeIndex(df.index).sort_values().unique()

        feed = fetch_yahoo_rss(ticker)
        if getattr(feed, "entries", None):
            if debug_mode:
                st.sidebar.subheader("Raw Feed Debug")
                st.sidebar.write(f"Total articles: {len(feed.entries)}")
                st.sidebar.write("First article:", feed.entries[0])

        sentiment_full, articles_from_stock_section = make_sentiment_dataframe(
            feed, keyword, trading_dates, pipe, debug_mode=debug_mode
        )

        if debug_mode:
            st.sidebar.subheader("Stock Data Debug")
            st.sidebar.write("Columns:", list(df.columns))
            st.sidebar.write("Shape:", df.shape)
            st.sidebar.write("Latest rows:", df.tail())
            st.sidebar.subheader("Sentiment Debug")
            st.sidebar.write("Matched sentiment rows:", len(articles_from_stock_section))
            st.sidebar.write("Processed sentiment:", sentiment_full.head())

        st.success("Successfully retrieved historical stock data")

        with st.expander("Combined Sentiment & Returns Analysis", expanded=False):
            st.write("### Combined Sentiment & Returns Analysis")
            combined_df = pd.DataFrame(
                {
                    "Sentiment Score": sentiment_full["Score"].reindex(df.index).fillna(0.0),
                    "Daily Return %": df["Daily Return %"],
                },
                index=df.index,
            )
            st.dataframe(
                style_with_colors(
                    combined_df,
                    {"Sentiment Score": "{:.2f}", "Daily Return %": "{:.2f}%"},
                    subset=["Sentiment Score", "Daily Return %"],
                ),
                height=400,
                use_container_width=True,
            )

        format_dict = {
            "Open": "${:.2f}",
            "High": "${:.2f}",
            "Low": "${:.2f}",
            "Close": "${:.2f}",
            "Adj Close": "${:.2f}",
            "Close_MA_20": "${:.2f}",
            "Close_MA_50": "${:.2f}",
            "Close_MA_100": "${:.2f}",
            "Volume": "{:,}",
            "Daily Return %": "{:.2f}%",
            "Return_MA_20": "{:.2f}%",
            "Return_MA_50": "{:.2f}%",
            "Return_MA_100": "{:.2f}%",
        }
        format_dict = {col: fmt for col, fmt in format_dict.items() if col in df.columns}

        with st.expander("Complete Historical Data", expanded=False):
            st.write("### Complete Historical Data")
            st.dataframe(df.style.format(format_dict, na_rep="-"), height=600, use_container_width=True)

        st.write("## Technical Analysis")

        st.write("### Price Moving Averages")
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(df.index, df["Close"], label="Closing Price", alpha=0.5)
        ax1.plot(df.index, df["Close_MA_20"], label="20-Day MA", linewidth=1.5)
        ax1.plot(df.index, df["Close_MA_50"], label="50-Day MA", linewidth=1.5)
        ax1.plot(df.index, df["Close_MA_100"], label="100-Day MA", linewidth=1.5)
        ax1.set_title(f"{ticker} Price Movement with Moving Averages")
        ax1.set_ylabel("Price (USD)")
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)

        st.write("### Daily Return Moving Averages")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(df.index, df["Daily Return %"], label="Daily Returns", alpha=0.3)
        ax2.plot(df.index, df["Return_MA_20"], label="20-Day MA", linewidth=1.5)
        ax2.plot(df.index, df["Return_MA_50"], label="50-Day MA", linewidth=1.5)
        ax2.plot(df.index, df["Return_MA_100"], label="100-Day MA", linewidth=1.5)
        ax2.set_title(f"{ticker} Daily Returns with Moving Averages")
        ax2.set_ylabel("Percentage Change")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        st.write("## Sentiment-Driven Return Forecasting")
        if len(df) < 80:
            st.warning("Insufficient data for forecasting. Need enough history to create lagged features and a test set.")
        else:
            try:
                build_forecast(df, sentiment_full, forecast_days, ticker)
            except Exception as e:
                st.error(f"Forecasting error: {e}")

# -----------------------------
# News Sentiment Analysis Section
# -----------------------------
st.write("## Complete News Sentiment Analysis")
try:
    feed = fetch_yahoo_rss(ticker)
    if not getattr(feed, "entries", None):
        st.warning("No articles found in RSS feed")
    else:
        articles = []
        sentiment_scores = []
        keyword_clean = keyword.lower()

        for entry in feed.entries:
            text = get_entry_text(entry)
            title = get_entry_title(entry)
            if keyword_clean and keyword_clean not in text.lower() and keyword_clean not in title.lower():
                continue
            try:
                label, score = sentiment_score(text, pipe)
                articles.append(
                    {
                        "Date": entry.get("published", "N/A"),
                        "Title": title,
                        "Sentiment": label,
                        "Score": score,
                        "Link": entry.get("link", ""),
                        "Full Text": text,
                    }
                )
                sentiment_scores.append(score)
            except Exception as e:
                st.error(f"Error processing article: {e}")

        if articles:
            st.write(f"### All Analyzed Articles ({len(articles)} total)")
            for idx, article in enumerate(articles, 1):
                with st.expander(f"Article {idx}: {article['Title']}"):
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(f"**Date:** {article['Date']}")
                        st.write(f"**Sentiment:** {article['Sentiment']}")
                        st.write(f"**Score:** {article['Score']:.2f}")
                        if article["Link"]:
                            st.write(f"[Read Full Article]({article['Link']})")
                    with col2:
                        st.write("**Summary:**")
                        st.write(article["Full Text"])

            st.write("### Aggregate Sentiment Analysis")
            avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            positive_count = len([s for s in sentiment_scores if s > 0])
            negative_count = len([s for s in sentiment_scores if s < 0])

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Articles Analyzed", len(articles))
            col2.metric("Average Sentiment Score", f"{avg_score:.2f}")
            col3.metric("Positive/Negative Ratio", f"{positive_count}:{negative_count}")
        else:
            st.warning("No articles matched the keyword filter")
except Exception as e:
    st.error(f"News analysis error: {e}")

# -----------------------------
# Conclusion Section
# -----------------------------
st.write("## Analysis Conclusion")

if not stock_data_available:
    st.info("Conclusion is unavailable because stock data was not successfully retrieved.")
else:
    try:
        last_close = float(df["Close"].iloc[-1])
        min_close = float(df["Close"].tail(5).min())
        max_close = float(df["Close"].tail(5).max())
        recent_volatility = float(df["Daily Return %"].tail(5).std())
        ma_50 = float(df["Close_MA_50"].iloc[-1]) if not pd.isna(df["Close_MA_50"].iloc[-1]) else np.nan

        support_resistance = f"${min_close:.2f}-${max_close:.2f}" if min_close != max_close else "N/A"
        position_signal = "long" if not pd.isna(ma_50) and last_close > ma_50 else "short/watchlist"

        conclusion = f"""
**Key Insights for {ticker}:**
- **Final Closing Price**: ${last_close:.2f}
- **Support/Resistance Levels**: {support_resistance}
- **Recent Volatility**: {recent_volatility:.2f}% (5-day STD)

**Recommendations:**
1. Monitor price action around key level: {support_resistance}
2. Consider **{position_signal}** positioning based on the 50-Day MA signal
3. Review upcoming earnings dates and news sentiment trends for confirmation
"""
        st.markdown(conclusion)
        st.success("Analysis completed! Enter a new ticker and click Run Analysis again.")
    except Exception as e:
        st.error(f"Could not build conclusion: {e}")

# -----------------------------
# Methodology Section
# -----------------------------
with st.expander("Methodology Notes", expanded=False):
    st.write("---")
    st.write(
        """
    **Methodology Notes:**
    - **Data Sources:**
        - Stock data sourced from Yahoo Finance historical records through `yfinance`
        - News sentiment derived from Yahoo Finance RSS feed articles
        - External economic indicators are not currently included

    - **Technical Analysis:**
        - Daily returns calculated using closing prices
        - Price moving averages calculated on closing prices (20/50/100-Day SMA)
        - Return moving averages calculated on daily percentage changes

    - **Sentiment Analysis:**
        - FinBERT processes article text with truncation to avoid model token-limit errors
        - Sentiment scores range from approximately -1 to +1
        - Articles are mapped to the same or next available trading day
        - Missing sentiment values are filled with 0 as a neutral baseline

    - **Machine Learning Forecasting:**
        - XGBoost regression model uses historical returns and sentiment lags
        - Time-series aware train/test split uses the first 80% for training and the final 20% for testing
        - Recursive multi-step forecasting is used for the selected forecast window
        - Forecast intervals are simple directional ranges, not statistically rigorous confidence intervals

    - **Limitations:**
        - Yahoo Finance may rate-limit requests, especially after repeated reruns
        - Sentiment data is limited to Yahoo Finance RSS articles
        - This app is for research/education and should not be treated as financial advice
    """
    )
