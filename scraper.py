import re
import os
import shelve
from urllib.parse import urlparse, urljoin, urldefrag, urlunparse, parse_qs, urlencode
from bs4 import BeautifulSoup
from utils import get_logger, get_urlhash

logger = get_logger("SCRAPER", "SCRAPER")

# ------------------------------------------------------------------
# Stop Words and Word Analysis for Reporting (Longest page, common words)
# ------------------------------------------------------------------
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
    file_exists = os.path.exists("words.shelve")
    with lock:
        with shelve.open("words.shelve") as word_store:
            if not file_exists:
                word_store['*'] = (0, None)  # (max_word_count, url)

    text = soup.get_text(separator=" ", strip=True).lower()
    english_text = re.sub(r"[^a-z\s]", "", text)
    words = english_text.split()
    words = [word for word in words if len(word) > 2]

    num_words = len(words)
    if num_words < limit:
        return 0

    counts = {}
    for word in words:
        if word in stop_words:
            continue
        counts[word] = counts.get(word, 0) + 1

    with lock:
        with shelve.open("words.shelve") as word_store:
            for word, count in counts.items():
                if word not in word_store:
                    word_store[word] = 0
                word_store[word] += count
            # Track the longest page seen so far
            if num_words > word_store['*'][0]:
                word_store['*'] = (num_words, url)
    return num_words

# ------------------------------------------------------------------
# Simhash Functions for Duplicate Detection
# ------------------------------------------------------------------
def compute_simhash(soup, hashbits=64):
    """
    Computes a 64-bit simhash fingerprint from the page text.
    This implementation tokenizes the text and builds a weighted fingerprint.
    """
    text = soup.get_text()
    tokens = text.split()
    freq = {}
    for token in tokens:
        token = token.lower()
        freq[token] = freq.get(token, 0) + 1

    v = [0] * hashbits
    for token, weight in freq.items():
        # Use Python's built-in hash; mask to 64 bits.
        h = hash(token) & ((1 << hashbits) - 1)
        for i in range(hashbits):
            bitmask = 1 << i
            if h & bitmask:
                v[i] += weight
            else:
                v[i] -= weight

    fingerprint = 0
    for i in range(hashbits):
        if v[i] >= 0:
            fingerprint |= (1 << i)
    return fingerprint

def hamming_distance(x, y):
    """Returns the Hamming distance between two integers."""
    return bin(x ^ y).count("1")

def is_duplicate(simhash1, simhash2, threshold=3):
    """Returns True if the Hamming distance between two simhashes is within the threshold."""
    return hamming_distance(simhash1, simhash2) <= threshold

def update_duplicate_cache(url, simhash_val, lock, cache_file="dupe_cache.shelve"):
    """
    Checks a shelve-based cache (with thread safety) to see if a page's simhash
    is near-duplicate of any previously seen page. If no duplicate is found, it stores the simhash.
    """
    with lock:
        with shelve.open(cache_file) as cache:
            for key, stored_simhash in cache.items():
                if is_duplicate(simhash_val, stored_simhash):
                    return True
            normalized = get_urlhash(url)
            cache[normalized] = simhash_val
    return False

# ------------------------------------------------------------------
# URL and Link Extraction Helpers
# ------------------------------------------------------------------
def allowed_url(url):
    """
    Returns True if the URL:
      - Uses http or https.
      - Belongs to an allowed UCI domain.
      - Does not have a disallowed file extension.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
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

def extract_links(soup, base_url):
    """
    Extracts all hyperlinks from the page.
    Converts relative URLs to absolute using the base URL and defragments them.
    """
    links = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        # Skip non-HTTP schemes
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        abs_url = urljoin(base_url, href)
        clean_url, _ = urldefrag(abs_url)
        links.append(clean_url)
    return links

# ------------------------------------------------------------------
# Main Scraper Function
# ------------------------------------------------------------------
def scraper(url, resp, lock):
    """
    Main scraper function.
    
    Process:
      1. Verify response status and that content is HTML.
      2. Parse the HTML with BeautifulSoup.
      3. Honor meta robots directives (noindex/nofollow).
      4. Compute a simhash fingerprint and use it to avoid near-duplicate pages.
      5. Update word counts for reporting (longest page, common words).
      6. Extract and filter outbound links (normalize URLs by removing fragments).
    
    The data collected (unique pages, longest page, word frequencies, subdomain counts)
    will be used to generate the assignment report.
    """
    if resp.status != 200:
        return []

    content = resp.raw_response.content
    if not content or b"<html" not in content.lower():
        return []

    soup = BeautifulSoup(content, "html.parser")

    # Honor meta robots: skip page if "noindex"; if "nofollow", do not extract links.
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

    # Compute simhash for duplicate detection
    simhash_val = compute_simhash(soup)
    if update_duplicate_cache(url, simhash_val, lock):
        logger.info(f"Duplicate page detected: {url}")
        return []

    # Update word statistics (for longest page and common words report)
    page_word_count = getWords(soup, lock, url)
    logger.info(f"Processed {url} with {page_word_count} words.")

    # Extract outbound links and filter them using allowed_url criteria.
    raw_links = extract_links(soup, resp.raw_response.url)
    valid_links = [link for link in raw_links if allowed_url(link)]
    
    return valid_links
