import re
import os
import shelve
import threading
import numpy as np
from urllib.parse import urlparse, urljoin, urldefrag, urlunparse, parse_qs, urlencode
from bs4 import BeautifulSoup
from utils import get_logger, get_urlhash
from simhash import EnhancedSimHash  # Import your simhash class
from collections import Counter

logger = get_logger("SCRAPER", "SCRAPER")
lock = threading.Lock()

stop_words = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could",
    "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes",
    "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im",
    "ive", "if", "in", "into", "is", "isnt", "it", "its", "itself", "lets", "me", "more", "most", "mustnt",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt",
    "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there",
    "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "were", "werent",
    "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys",
    "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself",
    "yourselves"
}

def getWords(soup, lock, url, limit=50):
    """
    Extracts text from the BeautifulSoup object, cleans it,
    counts word frequencies (ignoring stop words), and updates a shelve file for reporting.
    Also tracks the longest page (via a special key '*').
    Returns the total word count (if above a threshold) or 0.
    """
    # Ensure the shelve has been initialized with the special key '*'
    with lock, shelve.open("words.shelve") as word_store:
        if '*' not in word_store:
            word_store['*'] = (0, None)  # (max_word_count, url)
    
    # Extract and clean text
    raw_text = soup.get_text(separator=" ", strip=True).lower()
    cleaned_text = re.sub(r"[^a-z\s]", "", raw_text)
    tokens = [token for token in cleaned_text.split() if len(token) > 2]
    
    total_words = len(tokens)
    if total_words < limit:
        return 0

    # Count tokens that are not stop words
    token_counts = Counter(token for token in tokens if token not in stop_words)
    
    # Update the persistent shelve with token frequencies and the longest page info
    with lock, shelve.open("words.shelve") as word_store:
        for token, count in token_counts.items():
            word_store[token] = word_store.get(token, 0) + count
        if total_words > word_store['*'][0]:
            word_store['*'] = (total_words, url)
    
    return total_words

def update_duplicate_cache_np(url, simhash_val, lock, cache_file="dupe_cache.shelve", threshold=0.8):
    """
    Checks a shelve-based cache (with thread safety) for near-duplicate pages.
    The simhash_val is a numpy boolean array.
    Duplicate if similarity (1 - normalized Hamming distance) >= threshold.
    """
    with lock:
        with shelve.open(cache_file) as cache:
            for key, stored_simhash in cache.items():
                stored_array = np.array(stored_simhash, dtype=bool)
                hamming_dist = np.sum(simhash_val != stored_array)
                similarity = 1.0 - (hamming_dist / simhash_val.size)
                if similarity >= threshold:
                    return True
            normalized = get_urlhash(url)
            cache[normalized] = simhash_val.tolist()
    return False

def extract_links(soup, base_url):
    """
    Extracts all hyperlinks from the page.
    Converts relative URLs to absolute using the base URL and defragments them.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        abs_url = urljoin(base_url, href)
        clean_url, _ = urldefrag(abs_url)
        links.append(clean_url)
    return links

def scraper(url, resp):
    """
    Entry point for the crawler.
    Calls extract_next_links and then filters the returned URLs using is_valid.
    """
    links = extract_next_links(url, resp)
    return [link for link in links if is_valid(link)]

def extract_next_links(url, resp):
    """
    Processes the response and extracts outbound hyperlinks.
    
    Steps:
      1. Verify response status and HTML content.
      2. Parse HTML using BeautifulSoup.
      3. Honor meta robots directives: if "noindex" is present, skip the page;
         if "nofollow" is present, update word statistics but do not extract links.
      4. Compute a simhash fingerprint (using EnhancedSimHash from simhash.py) and check for duplicates.
      5. Update word analysis (for reporting: longest page, common words).
      6. Extract outbound links (defragmented and absolute).
    Returns a list of candidate hyperlinks.
    """
    if resp.status != 200:
        return []
    
    content = resp.raw_response.content
    if not content or b"<html" not in content.lower():
        return []
    
    soup = BeautifulSoup(content, "html.parser")
    
    # Honor meta robots: skip page if "noindex"; if "nofollow", update words only.
    meta_robots = soup.find("meta", attrs={"name": "robots"})
    if meta_robots:
        robots_content = meta_robots.get("content", "").lower()
        if "noindex" in robots_content:
            logger.info(f"Skipping {url} due to noindex directive.")
            return []
        if "nofollow" in robots_content:
            logger.info(f"Nofollow directive found for {url}; updating words only.")
            getWords(soup, lock, url)
            return []
    
    # Duplicate detection using EnhancedSimHash from simhash.py
    # Decode content to string for the simhash function.
    html_str = content.decode("utf-8", errors="ignore")
    simhash_instance = EnhancedSimHash()
    doc_simhash = simhash_instance.compute_document_hash(html_str)
    if update_duplicate_cache_np(url, doc_simhash, lock, threshold=simhash_instance.threshold):
        logger.info(f"Duplicate page detected: {url}")
        return []
    
    # Update word statistics (for longest page and common words report)
    page_word_count = getWords(soup, lock, url)
    logger.info(f"Processed {url} with {page_word_count} words.")
    
    # Extract outbound links
    raw_links = extract_links(soup, resp.raw_response.url)
    return raw_links

def is_valid(url):
    """
    Decide whether to crawl this URL or not.
    Uniqueness is based solely on the URL (ignoring fragments).
    
    Conditions:
      - URL must use http or https.
      - URL must belong to one of the allowed UCI domains.
      - URL must not point to unwanted file types (based on file extension).
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            return False

        allowed_domains = ("ics.uci.edu", "cs.uci.edu", "informatics.uci.edu", "stat.uci.edu")
        if not any(parsed.netloc.endswith(domain) for domain in allowed_domains):
            return False

        ext_pattern = (
            r".*\.(css|js|bmp|gif|jpe?g|ico"
            r"|png|tiff?|mid|mp2|mp3|mp4"
            r"|wav|avi|mov|mpeg|ram|m4v|mkv|ogg|ogv|pdf"
            r"|ps|eps|tex|ppt|pptx|doc|docx|xls|xlsx|names"
            r"|data|dat|exe|bz2|tar|msi|bin|7z|psd|dmg|iso"
            r"|epub|dll|cnf|tgz|sha1"
            r"|thmx|mso|arff|rtf|jar|csv"
            r"|rm|smil|wmv|swf|wma|zip|rar|gz)$"
        )
        if re.match(ext_pattern, parsed.path.lower()):
            return False

        return True
    except TypeError:
        print("TypeError for ", parsed)
        raise
