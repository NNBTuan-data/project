try:
    from duckduckgo_search import DDGS
    print("Imported DDGS from duckduckgo_search")
except ImportError:
    print("Could not import DDGS from duckduckgo_search")

try:
    from ddgs import DDGS
    print("Imported DDGS from ddgs")
except ImportError as e:
    print(f"Could not import DDGS from ddgs: {e}")

# Try to use it
import logging
logging.basicConfig(level=logging.DEBUG)

if 'DDGS' in locals():
    print("Testing DDGS...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images("chair", max_results=5))
        print(f"Found {len(results)} results")
    except Exception as e:
        print(f"Error: {e}")
