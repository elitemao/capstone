import requests
from bs4 import BeautifulSoup

def get_article(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    def extract_article_with_headings(soup: BeautifulSoup) -> str:
        """Fallback extractor: keep headings + paragraphs + list items."""
        article_el = soup.find("article") or soup
        parts = []
        for el in article_el.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
            t = el.get_text(" ", strip=True)
            if t:
                parts.append(t)
        return "\n\n".join(parts).strip()

    # --- Try finnish_media_scrapers first ---
    try:
        from finnish_media_scrapers.htmltotext import extract_text_from_yle_html
        text = extract_text_from_yle_html(html)
        if text and text.strip():
            # For Selkouutiset, headings can get flattened,
            # so prefer structured fallback if it finds something good:
            structured = extract_article_with_headings(soup)
            if structured:
                text = structured
            return {"content": text.strip()}
    except Exception:
        pass

    # --- Fallback: headings + paragraphs from HTML ---
    text = extract_article_with_headings(soup)
    if not text:
        raise ValueError("Could not extract article content (layout may have changed).")

    return {"content": text}
