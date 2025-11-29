import requests
from bs4 import BeautifulSoup

def get_article(url: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    html = r.text
    soup = BeautifulSoup(html, "html.parser")

    def title_from_text(text: str):
        if not text:
            return None
        cleaned = text.strip()
        prefixes = [
            "Selkouutiset Article text:",
            "Selkouutiset article text:",
            "Selkouutiset:",
        ]
        for p in prefixes:
            if cleaned.startswith(p):
                cleaned = cleaned[len(p):].strip()
                break
        first_line = cleaned.splitlines()[0].strip() if cleaned.splitlines() else ""
        if first_line.count(".") >= 2:
            return first_line
        if 10 <= len(first_line) <= 160:
            return first_line
        return None

    def extract_article_with_headings(soup: BeautifulSoup) -> str:
        article_el = soup.find("article") or soup
        parts = []
        for el in article_el.find_all(["h2", "h3", "p", "li"], recursive=True):
            t = el.get_text(" ", strip=True)
            if not t:
                continue
            parts.append(t)
        return "\n\n".join(parts).strip()

    # --- Try finnish_media_scrapers first ---
    try:
        from finnish_media_scrapers.htmltotext import extract_text_from_yle_html
        text = extract_text_from_yle_html(html)
        if text and text.strip():
            title = title_from_text(text)
            if not title:
                h1 = soup.find("h1")
                title = h1.get_text(" ", strip=True) if h1 else ""
            # BUT: override text for Selkouutiset pages to keep headings
            # (scraper sometimes flattens/loses them)
            text2 = extract_article_with_headings(soup)
            if text2:
                text = text2
            return {"title": title, "text": text.strip()}
    except Exception:
        pass

    # --- Fallback: headings + paragraphs from HTML ---
    h1 = soup.find("h1")
    h1_title = h1.get_text(" ", strip=True) if h1 else ""

    text = extract_article_with_headings(soup)
    if not text:
        raise ValueError("Could not extract article text (layout may have changed).")

    title = title_from_text(text) or h1_title
    return {"title": title, "text": text}



# ✅ only runs when you do: python fetch_yle_article.py
#if __name__ == "__main__":
#    data = get_article("https://yle.fi/a/74-20194402")
#    print(data["title"])
#    print(data["text"][:1200])