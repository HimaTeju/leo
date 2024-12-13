import streamlit as st
import pandas as pd
from alpha_vantage.fundamentaldata import FundamentalData

# Set up your Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = 'CY46Z49NI9DXRMMS'

# Initialize the Alpha Vantage API client
fd = FundamentalData(ALPHA_VANTAGE_API_KEY)

def fetch_fundamental_data(symbol):
    try:
        data, _ = fd.get_company_overview(symbol)
        
        pe_ratio = data.get('PERatio', 'N/A')
        eps = data.get('EPS', 'N/A')
        market_cap = data.get('MarketCapitalization', 'N/A')
        sector = data.get('Sector', 'N/A')
        industry = data.get('Industry', 'N/A')
        
        return {
            'P/E Ratio': pe_ratio,
            'EPS': eps,
            'Market Cap': market_cap,
            'Sector': sector,
            'Industry': industry
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return {}

def calculate_investment_decision(pe_ratio, eps, market_cap):
    # Default values
    recommendation = "No recommendation"
    pe_status = "N/A"
    eps_status = "N/A"
    market_cap_status = "N/A"
    investment_color = "yellow"  # Default to neutral

    # P/E Ratio Analysis
    if pe_ratio != 'N/A':
        try:
            pe_ratio = float(pe_ratio)
            if pe_ratio < 15:
                pe_status = "Undervalued"
            elif 15 <= pe_ratio <= 25:
                pe_status = "Fairly Priced"
            else:
                pe_status = "Overvalued"
        except ValueError:
            pe_status = "Invalid P/E Ratio"

    # EPS Analysis
    if eps != 'N/A':
        try:
            eps = float(eps)
            if eps < 0:
                eps_status = "Negative EPS"
            else:
                eps_status = "Positive EPS"
        except ValueError:
            eps_status = "Invalid EPS"

    # Market Cap Analysis
    if market_cap != 'N/A':
        try:
            market_cap = float(market_cap)
            if market_cap > 100000000000:  # Large-cap
                market_cap_status = "Large Cap"
            elif market_cap > 10000000000:  # Mid-cap
                market_cap_status = "Mid Cap"
            else:  # Small-cap
                market_cap_status = "Small Cap"
        except ValueError:
            market_cap_status = "Invalid Market Cap"
    
    # Final Recommendation Logic based on PE, EPS, and Market Cap
    if pe_status == "Overvalued" or eps_status == "Negative EPS" or market_cap_status == "Small Cap":
        investment_color = "red"
        recommendation = "Consider avoiding"
    elif pe_status == "Undervalued" and eps_status == "Positive EPS" and market_cap_status == "Large Cap":
        investment_color = "green"
        recommendation = "Strong buy"
    
    return recommendation, pe_status, eps_status, market_cap_status, investment_color

# Streamlit Frontend
def main():
    st.title('Stock Fundamental Analysis & Comparison Tool')

    st.sidebar.header('Input Details')

    symbol_1 = st.sidebar.text_input('Enter First Company Symbol (e.g., TCS, INFY, etc.)', 'TCS')
    symbol_2 = st.sidebar.text_input('Enter Second Company Symbol (e.g., TCS, INFY, etc.)', 'INFY')

    if symbol_1 and symbol_2:
        st.subheader(f"Fetching data for {symbol_1} and {symbol_2}...")

        # Fetching data for both companies
        fundamentals_1 = fetch_fundamental_data(symbol_1)
        fundamentals_2 = fetch_fundamental_data(symbol_2)

        if fundamentals_1 and fundamentals_2:
            # Displaying fundamental analysis for both stocks
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Stock 1: " + symbol_1)
                st.write(f"**P/E Ratio**: {fundamentals_1.get('P/E Ratio')}")
                st.write(f"**EPS**: {fundamentals_1.get('EPS')}")
                st.write(f"**Market Cap**: {fundamentals_1.get('Market Cap')}")
                st.write(f"**Sector**: {fundamentals_1.get('Sector')}")
                st.write(f"**Industry**: {fundamentals_1.get('Industry')}")
                
            with col2:
                st.markdown("### Stock 2: " + symbol_2)
                st.write(f"**P/E Ratio**: {fundamentals_2.get('P/E Ratio')}")
                st.write(f"**EPS**: {fundamentals_2.get('EPS')}")
                st.write(f"**Market Cap**: {fundamentals_2.get('Market Cap')}")
                st.write(f"**Sector**: {fundamentals_2.get('Sector')}")
                st.write(f"**Industry**: {fundamentals_2.get('Industry')}")

            # Getting the investment recommendation for both stocks
            recommendation_1, pe_status_1, eps_status_1, market_cap_status_1, investment_color_1 = calculate_investment_decision(
                fundamentals_1.get('P/E Ratio'),
                fundamentals_1.get('EPS'),
                fundamentals_1.get('Market Cap')
            )
            recommendation_2, pe_status_2, eps_status_2, market_cap_status_2, investment_color_2 = calculate_investment_decision(
                fundamentals_2.get('P/E Ratio'),
                fundamentals_2.get('EPS'),
                fundamentals_2.get('Market Cap')
            )

            # Displaying the comparison result for both stocks
            st.markdown(f"### Investment Recommendation Comparison:")
            
            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"**{symbol_1} Recommendation**")
                st.write(f"**P/E Status**: {pe_status_1}")
                st.write(f"**EPS Status**: {eps_status_1}")
                st.write(f"**Market Cap Status**: {market_cap_status_1}")
                st.markdown(f'<h3 style="color:{investment_color_1}">{recommendation_1}</h3>', unsafe_allow_html=True)
                
            with col4:
                st.markdown(f"**{symbol_2} Recommendation**")
                st.write(f"**P/E Status**: {pe_status_2}")
                st.write(f"**EPS Status**: {eps_status_2}")
                st.write(f"**Market Cap Status**: {market_cap_status_2}")
                st.markdown(f'<h3 style="color:{investment_color_2}">{recommendation_2}</h3>', unsafe_allow_html=True)

        else:
            st.write("No data found for one or both stocks. Please try again.")
    else:
        st.write("Please enter valid stock symbols.")

if __name__ == "__main__":
    main()
