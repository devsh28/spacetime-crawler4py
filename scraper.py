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

# -----------------------------
# Word Analysis Function
# -----------------------------
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

# -----------------------------
# Duplicate Detection Using Simhash (via numpy)
# -----------------------------
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

# -----------------------------
# URL Filtering and Link Extraction
# -----------------------------
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

# -----------------------------
# Subdomain Analysis Functions
# -----------------------------
def update_subdomain_count(url, lock, subdomain_file="subdomains.shelve"):
    """
    Updates a shelve-based store for subdomain analysis.
    For URLs under the ics.uci.edu domain, stores the unique URL (ignoring fragments)
    under its subdomain (the netloc). This will be used to count unique pages per subdomain.
    """
    parsed = urlparse(url)
    if parsed.netloc.lower().endswith("ics.uci.edu"):
        subdomain = parsed.netloc.lower()
        with lock, shelve.open(subdomain_file) as subdomains:
            if subdomain in subdomains:
                urls_set = subdomains[subdomain]
                urls_set.add(url)
                subdomains[subdomain] = urls_set
            else:
                subdomains[subdomain] = {url}

def analyze_subdomains(lock, subdomain_file="subdomains.shelve"):
    """
    Returns a dictionary mapping each subdomain (from ics.uci.edu) to the number of unique pages.
    """
    with lock, shelve.open(subdomain_file) as subdomains:
        return {subdomain: len(urls) for subdomain, urls in subdomains.items()}

# -----------------------------
# Infinite Trap Detection
# -----------------------------
def infinite_trap_detect(url):
    """
    Checks for potential infinite crawler traps based on repeated path segments
    and date patterns in the path or query.
    Returns True if the URL appears to be an infinite trap.
    """
    parsed = urlparse(url)
    
    # Check for repeated path segments
    path_segments = [seg for seg in parsed.path.split("/") if seg]
    segment_counts = {}
    for seg in path_segments:
        segment_counts[seg] = segment_counts.get(seg, 0) + 1
        if segment_counts[seg] > 2:
            return True

    # Check for date patterns that may indicate calendar traps
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", parsed.query) is not None:
        return True
    if re.search(r"\b\d{4}-\d{2}\b", parsed.query) is not None:
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", parsed.path) is not None:
        return True
    if re.search(r"\b\d{4}-\d{2}\b", parsed.path) is not None:
        return True

    return False

# -----------------------------
# Required Function Signatures
# -----------------------------
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
    
    Args:
        url (str): The URL that was crawled
        resp (Response): Response object containing status and content
    
    Returns:
        list: List of valid URLs extracted from the page
    """
    # 1. Initial response validation
    if resp is None:
        logger.error(f"No response object for URL: {url}")
        return []
        
    if resp.status != 200:
        logger.error(f"Bad status code {resp.status} for URL: {url}")
        return []
        
    if resp.raw_response is None:
        logger.error(f"No raw response for URL: {url}")
        return []

    # 2. Content validation
    try:
        content = resp.raw_response.content
        if not content:
            logger.error(f"Empty content for URL: {url}")
            return []
            
        # Check if content is HTML
        if not is_html(content):
            logger.info(f"Skipping non-HTML content at {url}")
            return []
            
    except Exception as e:
        logger.error(f"Error accessing content for {url}: {e}")
        return []

    # 3. Parse HTML
    try:
        soup = BeautifulSoup(content, "html.parser", from_encoding="utf-8")
    except Exception as e:
        logger.error(f"BeautifulSoup parsing error for {url}: {e}")
        return []

    # 4. Check robots meta tags
    try:
        if meta_robots := soup.find("meta", attrs={"name": "robots"}):
            robots_content = meta_robots.get("content", "").lower()
            if "noindex" in robots_content:
                logger.info(f"Skipping {url} due to noindex directive")
                return []
            if "nofollow" in robots_content:
                logger.info(f"Nofollow directive found for {url}; updating words only")
                getWords(soup, lock, url)
                return []
    except Exception as e:
        logger.error(f"Error checking robots meta for {url}: {e}")

    # 5. Check for infinite traps
    if infinite_trap_detect(url):
        logger.info(f"Skipping {url} due to potential infinite trap")
        return []

    # 6. Extract and process links
    links = []
    try:
        for a_tag in soup.find_all("a", href=True):
            href = a_tag.get("href", "").strip()
            
            # Skip empty or javascript links
            if not href or href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
                continue

            # Convert relative to absolute URL
            try:
                absolute_url = urljoin(resp.raw_response.url, href)
                # Remove fragments
                clean_url, _ = urldefrag(absolute_url)
                
                # Basic URL validation
                if not clean_url.startswith(('http://', 'https://')):
                    continue
                    
                # Add valid URL to list
                if clean_url != url:  # Avoid self-links
                    links.append(clean_url)
                    
            except Exception as e:
                logger.error(f"Error processing link {href} from {url}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error extracting links from {url}: {e}")
        return []

    # 7. Process page content for word analysis
    try:
        page_word_count = getWords(soup, lock, url)
        logger.info(f"Processed {url} with {page_word_count} words")
    except Exception as e:
        logger.error(f"Error processing words for {url}: {e}")

    # 8. Update subdomain statistics
    try:
        update_subdomain_count(url, lock)
    except Exception as e:
        logger.error(f"Error updating subdomain count for {url}: {e}")

    # 9. Log summary
    logger.info(f"Extracted {len(links)} links from {url}")
    
    return links

def is_html(content):
    """
    Check if content is HTML by looking for common HTML markers
    """
    try:
        # Check for PDF
        if content.startswith(b'%PDF'):
            return False
            
        # Look for HTML markers
        content_lower = content.lower()
        return (b'<html' in content_lower or 
                b'<head' in content_lower or 
                b'<body' in content_lower or 
                b'<!doctype html' in content_lower)
    except Exception:
        return False

def is_valid(url):
    """
    Decide whether to crawl this URL or not.
    Uniqueness is based solely on the URL (ignoring fragments) and avoiding traps.
    
    Conditions:
      - URL must use http or https.
      - URL must belong to one of the allowed UCI domains.
      - URL must not point to unwanted file types (based on file extension).
      - URL must not trigger infinite crawler traps.
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

        # Additional infinite tra n p detection: Check for repeated path segments
        path_segments = [seg for seg in parsed.path.split("/") if seg]
        segment_counts = {}
        for seg in path_segments:
            segment_counts[seg] = segment_counts.get(seg, 0) + 1
            if segment_counts[seg] > 2:
                return False

        # Include date pattern checks to avoid calendar traps:
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", parsed.query) is not None:
            return False
        if re.search(r"\b\d{4}-\d{2}\b", parsed.query) is not None:
            return False
        if re.search(r"\b\d{4}-\d{2}-\d{2}\b", parsed.path) is not None:
            return False
        if re.search(r"\b\d{4}-\d{2}\b", parsed.path) is not None:
            return False

        return True
    except TypeError:
        print("TypeError for ", parsed)
        raise

def write_report(report_filename="report.txt"):
    """
    Generates a report with:
      1. Total number of unique pages crawled.
      2. The longest page (by word count) with its URL.
      3. The 50 most common words with their counts.
      4. The subdomain analysis for pages under ics.uci.edu.
    The report is written to a text file.
    """
    # 1. Unique Pages: Count keys in the duplicate cache
    with shelve.open("cache.shelve") as cache:
        unique_count = len(cache)
    
    # 2. Longest Page: Read special key '*' from words.shelve
    with shelve.open("words.shelve") as word_store:
        longest_info = word_store.get('*', (0, "N/A"))
        longest_count, longest_url = longest_info

    # 3. 50 Most Common Words: Exclude the '*' key
    words_freq = {}
    with shelve.open("words.shelve") as word_store:
        for key in word_store:
            if key != '*':
                words_freq[key] = word_store[key]
    sorted_words = sorted(words_freq.items(), key=lambda x: x[1], reverse=True)
    top_50_words = sorted_words[:50]

    # 4. Subdomain Analysis: Count unique pages per subdomain from subdomains.shelve
    with shelve.open("subdomains.shelve") as subdomains:
        subdomain_dict = {k: len(v) for k, v in subdomains.items()}
    sorted_subdomains = sorted(subdomain_dict.items(), key=lambda x: x[0])
    subdomain_count = len(sorted_subdomains)

    # Write the report to a file
    with open(report_filename, "w") as f:
        f.write(f"1. {unique_count} Unique Pages Crawled\n")
        f.write("=" * 100 + "\n")
        f.write(f"2. Longest Page: {longest_url} with {longest_count} words\n")
        f.write("=" * 100 + "\n")
        f.write("3. 50 Most Common Words\n")
        f.write("-" * 23 + "\n")
        for word, freq in top_50_words:
            f.write(f"{word} ({freq})\n")
        f.write("=" * 100 + "\n")
        f.write(f"4. {subdomain_count} subdomains found\n")
        f.write("-" * 25 + "\n")
        for subdomain, count in sorted_subdomains:
            f.write(f"{subdomain}, {count}\n")

