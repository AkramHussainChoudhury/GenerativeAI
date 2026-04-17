# price_comparison_agent.py

from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from ddgs import DDGS
from playwright.sync_api import sync_playwright
from pydantic import BaseModel
import resend
from dotenv import load_dotenv
import os
import time


load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
resend.api_key = os.getenv("RESEND_API_KEY")




# ── Site name → domain mapping ──────────────────────────────
SITE_MAP = {
    "amazon":   {"domain": "amazon.in",    "product_path": "/dp/"},
    "flipkart": {"domain": "flipkart.com", "product_path": "/p/"},
    "myntra":   {"domain": "myntra.com",   "product_path": None},
    "meesho":   {"domain": "meesho.com",   "product_path": None},
}

# ── Pydantic schema for structured LLM output ───────────────
class PriceExtraction(BaseModel):
    product_name: str
    price: Optional[float] = None
    currency: str = "INR"

# ── Graph State ──────────────────────────────────────────────
class ProductState(TypedDict):
    product_name: str
    sites: List[str]
    target_price: float
    current_site_index: int
    current_url: Optional[str]
    current_price: Optional[float]
    results: Dict[str, Any]
    best_price: Optional[float]
    best_site: Optional[str]
    alert_needed: bool
    email_sent: bool

structured_llm = llm.with_structured_output(PriceExtraction)

# ── Node 1: Search ───────────────────────────────────────────
def search_node(state: ProductState) -> dict:
    product_name= state["product_name"]
    site_name=state["sites"][state["current_site_index"]].lower()
    site_info = SITE_MAP.get(site_name, {"domain": site_name, "product_path": None})
    domain = site_info["domain"]
    product_path = site_info["product_path"]

    print(f"\n[search_node] '{product_name}' on {site_name} ({domain})...")

    query = f'"{product_name}" site:{domain}'
    product_url=None

    try:
        with DDGS() as ddgs:
            results=list(ddgs.text(query, max_results=5, backend="html"))

        for r in results:
            url = r.get("href", "")
            if domain in url :
                if product_path and product_path not in url:
                    continue          # skip reviews, ads, search pages
                product_url = url
                break
    except Exception as e:
        print(f" [search_node] Error: {e}")

    print(f" [search_node] Found: {product_url}")
    time.sleep(1)
    return {"current_url" : product_url}


# ── Node 2: Fetch Price ──────────────────────────────────────
def fetch_price_node(state: ProductState) -> dict:
    url = state.get("current_url")

    if not url:
        print("  [fetch_price_node] No URL, skipping.")
        return {"current_price": None}

    print(f"  [fetch_price_node] Fetching {url}...")

    page_text = ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            page = browser.new_page(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="en-IN",
            )
            page.goto(url, timeout=30000, wait_until="load")
            time.sleep(3)
            page_text = page.inner_text("body")[:8000]
            browser.close()
    except Exception as e:
        print(f"  [fetch_price_node] Playwright error: {e}")
        return {"current_price": None}

    price = None
    for attempt in range(3):
        try:
            result = structured_llm.invoke([
                HumanMessage(content=(
                    f"Extract the product name and current selling price from this e-commerce page.\n"
                    f"Rules:\n"
                    f"- Return the current selling price (the price you pay today), NOT the MRP or crossed-out original price\n"
                    f"- Do NOT return EMI amounts or per-month prices\n"
                    f"- Return price as a plain number without currency symbols (e.g. 1299.0)\n"
                    f"- If no price is found, return null\n\n"
                    f"{page_text}"
                ))
            ])
            print(f"  [debug] LLM raw result (attempt {attempt+1}): {result}")
            price = result.price if result else None
            if price is not None:
                break
            time.sleep(2)
        except Exception as e:
            print(f"  [fetch_price_node] LLM error (attempt {attempt+1}): {e}")
            time.sleep(2)

    print(f"  [fetch_price_node] Price: {price}")
    return {"current_price": price}


# ── Node 3: Update Results ───────────────────────────────────
def update_results_node(state: ProductState) -> dict:
    idx = state["current_site_index"]
    site_name = state["sites"][idx].lower()

    results = dict(state.get("results",{}))
    results[site_name] = {
        "url": state.get("current_url"),
        "price": state.get("current_price"),
    }

    print(f"  [update_results_node] {site_name}: ₹{state.get('current_price')}")

    return {
        "results": results,
        "current_site_index": idx + 1,
        "current_url": None,
        "current_price": None,
    }

# ── Node 4: Compare Prices ───────────────────────────────────
def compare_prices_node(state: ProductState) -> dict:
    results = state.get("results", {})
    target_price = state["target_price"]

    best_price = None
    best_site = None

    print("\n[compare_prices_node] Price comparison:")
    for site, data in results.items():
        price = data.get("price")
        print(f"  {site:12s}: ₹{price}")
        if price is not None:
            if best_price is None or price < best_price:
                best_price = price
                best_site = site

    alert_needed = best_price is not None and best_price <= target_price

    print(f"  Best price : ₹{best_price} on {best_site}")
    print(f"  Target     : ₹{target_price}")
    print(f"  Alert?     : {alert_needed}")

    return {
        "best_price": best_price,
        "best_site": best_site,
        "alert_needed": alert_needed,
    }



# ── Node 5: Alert ────────────────────────────────────────────
def alert_node(state: ProductState) -> dict:
    product_name = state["product_name"]
    best_price = state["best_price"]
    best_site = state["best_site"]
    target_price = state["target_price"]
    results = state.get("results", {})
    best_url = results.get(best_site, {}).get("url", "N/A")

    rows = "\n".join(
        f"  {site:12s}: ₹{data['price']}"
        for site, data in results.items()
    )

    params = {
        "from": "Price Alert <onboarding@resend.dev>",
        "to": [os.getenv("EMAIL_RECEIVER")],
        "subject": f"Price Alert: {product_name} — ₹{best_price} on {best_site}",
        "html": f"""
            <h2>Price Drop Alert!</h2>
            <p><b>{product_name}</b> is now <b>₹{best_price}</b> on <b>{best_site}</b>,
            below your target of ₹{target_price}.</p>
            <h3>Price Comparison</h3>
            <pre>{rows}</pre>
            <p><a href="{best_url}">Buy Now</a></p>
        """,
    }

    try:
        resend.Emails.send(params)
        print(f"\n[alert_node] Email sent to {os.getenv('EMAIL_RECEIVER')}")
        return {"email_sent": True}
    except Exception as e:
        print(f"\n[alert_node] Email error: {e}")
        return {"email_sent": False}


def more_sites_to_check(state: ProductState) -> str:
    if state["current_site_index"] < len(state["sites"]):
        return "search"
    return "compare"

def should_send_alert(state: ProductState) -> str:
    return "alert" if state.get("alert_needed") else "end"



def build_graph():
    graph= StateGraph(ProductState)
    graph.add_node("search",search_node)
    graph.add_node("fetch_price",fetch_price_node)
    graph.add_node("update_results",update_results_node)
    graph.add_node("compare_prices",compare_prices_node)
    graph.add_node("alert",alert_node)


    graph.add_edge(START,"search")
    graph.add_edge("search","fetch_price")
    graph.add_edge("fetch_price","update_results")
    graph.add_conditional_edges(
        "update_results",
        more_sites_to_check,
        {"search": "search","compare":"compare_prices"},

    )

    graph.add_conditional_edges(
        "compare_prices",
        should_send_alert,
        {"alert":"alert","end":END}
    )

    graph.add_edge("alert",END)

    return graph.compile()


    # ── Entry Point ──────────────────────────────────────────────
if __name__ == "__main__":
    app = build_graph()

    print("=" * 55)
    print("Price Comparison Agent")
    print("=" * 55)
    product_name = input("Product name       : ").strip()
    sites_input  = input("Sites (e.g. amazon,flipkart,myntra,meesho): ").strip()
    target_price = float(input("Target price (₹)   : ").strip())

    sites = [s.strip().lower() for s in sites_input.split(",") if s.strip()]

    initial_state: ProductState = {
        "product_name": product_name,
        "sites": sites,
        "target_price": target_price,
        "current_site_index": 0,
        "current_url": None,
        "current_price": None,
        "results": {},
        "best_price": None,
        "best_site": None,
        "alert_needed": False,
        "email_sent": False,
    }

    print("=" * 55)
    print(f"Product     : {product_name}")
    print(f"Sites       : {', '.join(sites)}")
    print(f"Target Price: ₹{target_price}")
    print("=" * 55)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 55)
    print("FINAL RESULTS")
    print("=" * 55)
    for site, data in final_state["results"].items():
        price = data["price"]
        if price is not None:
            print(f"  {site:12s}: ₹{price}")
        else:
            print(f"  {site:12s}: Not found")

    best_price = final_state["best_price"]
    target_price = initial_state["target_price"]

    print(f"\n  Best Price  : ₹{best_price} on {final_state['best_site']}")
    if best_price is not None:
        savings = target_price - best_price
        if savings > 0:
            print(f"  Savings     : ₹{savings:.2f} below your target")
        else:
            print(f"  Above target: ₹{abs(savings):.2f} over your target")
    print(f"  Alert Sent  : {final_state['email_sent']}")










 

