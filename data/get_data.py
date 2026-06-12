"""
Data Acquisition Scripts for GRS
All data sources are publicly available and free.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta


def get_yahoo_esg_data(tickers, output_path='data/sample_esg_data.csv'):
    """
    Download ESG scores from Yahoo Finance.

    Yahoo Finance provides free ESG scores for major tickers.
    Coverage: S&P 500, NASDAQ-100, and other major indices.

    Parameters:
    -----------
    tickers : list
        List of stock tickers (e.g., ['AAPL', 'MSFT', 'TSLA'])
    output_path : str
        Path to save CSV

    Returns:
    --------
    pd.DataFrame with columns: ticker, esg_score, environment, social, governance
    """
    esg_data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            esg_data.append({
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'esg_score': info.get('esgScore', np.nan),
                'environment_score': info.get('environmentScore', np.nan),
                'social_score': info.get('socialScore', np.nan),
                'governance_score': info.get('governanceScore', np.nan),
                'market_cap': info.get('marketCap', np.nan),
                'date_fetched': datetime.now().strftime('%Y-%m-%d')
            })
            print(f"  ✓ {ticker}: ESG={info.get('esgScore', 'N/A')}")
        except Exception as e:
            print(f"  ✗ {ticker}: {str(e)}")
            esg_data.append({
                'ticker': ticker,
                'company_name': 'N/A',
                'sector': 'N/A',
                'industry': 'N/A',
                'esg_score': np.nan,
                'environment_score': np.nan,
                'social_score': np.nan,
                'governance_score': np.nan,
                'market_cap': np.nan,
                'date_fetched': datetime.now().strftime('%Y-%m-%d')
            })

    df = pd.DataFrame(esg_data)
    df.to_csv(output_path, index=False)
    print(f"
Saved: {output_path}")
    print(f"Total firms: {len(df)}")
    print(f"ESG data available: {df['esg_score'].notna().sum()}")
    return df


def get_sample_firm_list(n_firms=50):
    """
    Get sample list of firms with known ESG data on Yahoo Finance.

    These are major US firms with reliable ESG coverage.
    """
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
        'JPM', 'BAC', 'WFC', 'C', 'GS',           # Finance
        'XOM', 'CVX', 'COP', 'SLB', 'OXY',        # Energy
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',       # Healthcare
        'WMT', 'COST', 'HD', 'LOW', 'TGT',        # Consumer
        'PG', 'KO', 'PEP', 'MDLZ', 'GIS',         # Consumer staples
        'BA', 'CAT', 'GE', 'HON', 'MMM',          # Industrial
        'VZ', 'T', 'CMCSA', 'DIS', 'NFLX',        # Telecom/Media
        'NEE', 'DUK', 'SO', 'AEP', 'EXC'          # Utilities
    ]
    return sample_tickers[:n_firms]


def generate_sample_emdat_data(output_path='data/emdat_climate_shocks.csv', n_events=100):
    """
    Generate sample climate shock data based on EM-DAT format.

    NOTE: For production use, register at https://www.emdat.be/ and download
    the full dataset. This function generates representative sample data
    for demonstration and testing purposes.

    EM-DAT fields:
    - Year, Country, Dis No., Event, Location, Origin, Associated Dis
    - OFDA/BHA Response, Appeal, Declaration, Aid Contribution
    - Dis Mag Value, Dis Mag Scale, Latitude, Longitude, Local Time
    - River Basin, Start Year, Start Month, Start Day, End Year, End Month, End Day
    - Total Deaths, No. Injured, No. Affected, No. Homeless, Total Affected
    - Reconstruction Costs, Insured Damages, Total Damages, CPI
    """
    np.random.seed(42)

    countries = ['USA', 'China', 'Germany', 'France', 'UK', 'Japan', 'Australia', 
                 'Brazil', 'India', 'Canada', 'Italy', 'Spain', 'Netherlands']
    event_types = ['Drought', 'Flood', 'Storm', 'Wildfire', 'Extreme temperature']

    data = []
    for i in range(n_events):
        year = np.random.randint(2018, 2026)
        month = np.random.randint(1, 13)
        country = np.random.choice(countries)
        event_type = np.random.choice(event_types)

        # Severity correlates with damages
        severity = np.random.uniform(0.3, 1.0)
        total_damages = severity * np.random.uniform(1e6, 1e10)
        total_affected = severity * np.random.uniform(1e3, 1e6)

        data.append({
            'dis_no': f'{year}-{i:04d}',
            'year': year,
            'country': country,
            'event_type': event_type,
            'severity_index': round(severity, 3),
            'total_damages_usd': round(total_damages, 0),
            'total_affected': round(total_affected, 0),
            'start_month': month,
            'start_year': year,
            'latitude': np.random.uniform(-60, 70),
            'longitude': np.random.uniform(-180, 180),
            'data_source': 'EM-DAT_sample'
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Generated sample EM-DAT data: {output_path}")
    print(f"Events: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Event types: {df['event_type'].unique().tolist()}")
    return df


def generate_sample_network_data(output_path='data/network_edges.csv', n_firms=50):
    """
    Generate sample network edges based on sector and ownership patterns.

    In production, use SEC 13F filings from:
    https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F
    """
    np.random.seed(42)

    firms = [f'FIRM_{i:03d}' for i in range(n_firms)]
    sectors = ['Energy', 'Tech', 'Finance', 'Healthcare', 'Consumer']
    firm_sectors = {f: np.random.choice(sectors) for f in firms}

    edges = []

    # Sector edges (high weight)
    for i, f1 in enumerate(firms):
        for j, f2 in enumerate(firms[i+1:], i+1):
            if firm_sectors[f1] == firm_sectors[f2] and np.random.random() < 0.3:
                edges.append({
                    'source': f1,
                    'target': f2,
                    'layer': 'sector',
                    'weight': round(np.random.uniform(0.5, 1.0), 3)
                })

    # ESG provider edges (medium weight)
    providers = ['Sustainalytics', 'MSCI', 'Refinitiv']
    firm_providers = {f: np.random.choice(providers) for f in firms}
    for i, f1 in enumerate(firms):
        for j, f2 in enumerate(firms[i+1:], i+1):
            if firm_providers[f1] == firm_providers[f2] and np.random.random() < 0.2:
                edges.append({
                    'source': f1,
                    'target': f2,
                    'layer': 'esg_provider',
                    'weight': round(np.random.uniform(0.3, 0.8), 3)
                })

    # Ownership edges (low weight, random)
    n_institutions = 10
    for inst in range(n_institutions):
        holdings = np.random.choice(firms, size=np.random.randint(3, 15), replace=False)
        for i, f1 in enumerate(holdings):
            for f2 in holdings[i+1:]:
                if not any(e['source'] == f1 and e['target'] == f2 for e in edges):
                    edges.append({
                        'source': f1,
                        'target': f2,
                        'layer': 'ownership',
                        'weight': round(np.random.uniform(0.1, 0.5), 3)
                    })

    df = pd.DataFrame(edges)
    df.to_csv(output_path, index=False)
    print(f"Generated sample network data: {output_path}")
    print(f"Edges: {len(df)}")
    print(f"Layers: {df['layer'].unique().tolist()}")
    return df


def download_real_data():
    """
    Download real data from public sources.

    This function demonstrates how to acquire actual data for production use.
    """
    print("=" * 60)
    print("Downloading Real Public Data")
    print("=" * 60)

    # 1. Yahoo Finance ESG data (free, no API key needed)
    print("
[1] Yahoo Finance ESG Data")
    print("    Source: yfinance library (free)")
    print("    Coverage: Major US and international firms")
    tickers = get_sample_firm_list(n_firms=50)
    get_yahoo_esg_data(tickers, 'data/yahoo_esg_data.csv')

    # 2. EM-DAT (requires registration)
    print("
[2] EM-DAT Climate Disaster Data")
    print("    Source: https://www.emdat.be/")
    print("    Registration: Free academic registration required")
    print("    NOTE: Using sample data for demonstration")
    generate_sample_emdat_data('data/emdat_sample.csv', n_events=100)

    # 3. Network data (SEC 13F - public but requires parsing)
    print("
[3] SEC 13F Ownership Data")
    print("    Source: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F")
    print("    Access: Public, free, but requires parsing")
    print("    NOTE: Using sample data for demonstration")
    generate_sample_network_data('data/network_sample.csv', n_firms=50)

    print("
" + "=" * 60)
    print("Data download complete!")
    print("=" * 60)
    print("
For production use:")
    print("  1. Register at EM-DAT for full climate disaster data")
    print("  2. Parse SEC 13F filings for real ownership networks")
    print("  3. Yahoo Finance ESG data is ready to use")


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    download_real_data()
