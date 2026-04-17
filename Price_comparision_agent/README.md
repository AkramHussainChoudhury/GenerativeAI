# Price Comparison Alert Agent

An agentic AI system built with **LangGraph** that compares product prices across Indian e-commerce sites and sends an email alert when the price drops below your target.

No product URLs needed — just give it a product name and it finds, fetches, and compares prices automatically.

## How It Works

```
User Input (product name, sites, target price)
        │
        ▼
  search_node  ◄─────────────────────────────────┐
  DuckDuckGo discovers product URL                │ loops
        │                                         │ per site
        ▼                                         │
  fetch_price_node                                │
  Playwright fetches page + LLM extracts price    │
        │                                         │
        ▼                                         │
  update_results_node ── more sites? ─────────────┘
        │
        └── all done ──► compare_prices_node
                               │
                    price ≤ target? ──► alert_node ──► Email sent
                               │
                    price > target? ──► Done (no alert)
```

## Supported Sites

| Site     | Domain         |
|----------|----------------|
| amazon   | amazon.in      |
| flipkart | flipkart.com   |
| myntra   | myntra.com     |
| meesho   | meesho.com     |

## Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (free)
- A [Resend API key](https://resend.com/) (free tier works)

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/AkramHussainChoudhury/GenerativeAI.git
cd GenerativeAI
git checkout price_alert_agent
cd Price_comparision_agent
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Playwright browser

```bash
playwright install chromium
```

### 5. Set up environment variables

Create a `.env` file in the project folder:

```
GROQ_API_KEY=your_groq_api_key
RESEND_API_KEY=your_resend_api_key
EMAIL_RECEIVER=your_email@example.com
```

## Run

```bash
python price_comparison_agent.py
```

You will be prompted to enter:

```
========================================================
Price Comparison Agent
========================================================
Product name       : boAt Rockerz 450 Bluetooth Headphones
Sites (e.g. amazon,flipkart,myntra,meesho): amazon,flipkart
Target price (₹)   : 1500
```

## Example Output

```
========================================================
FINAL RESULTS
========================================================
  amazon      : ₹1,299
  flipkart    : ₹1,249
  myntra      : Not found
  meesho      : Not found

  Best Price  : ₹1,249 on flipkart
  Savings     : ₹251.00 below your target
  Alert Sent  : True
```

If the best price is at or below your target, an email alert is sent to your `EMAIL_RECEIVER` address with a price comparison table and a direct link to buy.

## Adding More Sites

Add an entry to `SITE_MAP` in `price_comparison_agent.py`:

```python
SITE_MAP = {
    "amazon":   {"domain": "amazon.in",    "product_path": "/dp/"},
    "flipkart": {"domain": "flipkart.com", "product_path": "/p/"},
    "myntra":   {"domain": "myntra.com",   "product_path": None},
    "meesho":   {"domain": "meesho.com",   "product_path": None},
    "croma":    {"domain": "croma.com",    "product_path": None},  # example
}
```

`product_path` filters out non-product pages (reviews, search results). Set to `None` if unknown.

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — agent graph orchestration
- [Groq](https://console.groq.com/) — LLM for price extraction (llama-3.3-70b)
- [Playwright](https://playwright.dev/python/) — headless browser for JS-rendered pages
- [DuckDuckGo Search](https://pypi.org/project/ddgs/) — product URL discovery
- [Resend](https://resend.com/) — email alerts
- [Pydantic](https://docs.pydantic.dev/) — structured LLM output
