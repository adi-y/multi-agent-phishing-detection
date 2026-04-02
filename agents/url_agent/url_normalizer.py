"""
URL Normalizer — Fixes Scheme & www. Bias
==========================================
Problem: Dataset has 91.7% bare URLs (no scheme, no www.)
         Real world input has https://, http://, www. prefixes
         This mismatch causes legitimate sites to be flagged

Solution: Normalize ALL URLs to a consistent format
          BEFORE feature extraction so training and
          real-world input are treated identically

Rules:
1. Strip www. from domain (www is not a signal)
2. Record scheme separately as features
3. Normalize to bare format for structural analysis
4. Add is_exact_brand_match to fix confusion score issue
"""

import re
from urllib.parse import urlparse


# Known legitimate exact brand domains
# These are EXACT matches — not impostors
LEGITIMATE_EXACT_DOMAINS = {
    "google.com", "gmail.com", "youtube.com", "google.co.in",
    "paypal.com", "paypal.co.uk",
    "facebook.com", "instagram.com", "whatsapp.com",
    "apple.com", "icloud.com",
    "microsoft.com", "outlook.com", "office.com", "live.com",
    "amazon.com", "amazon.in", "amazon.co.uk",
    "netflix.com", "twitter.com", "x.com",
    "linkedin.com", "reddit.com", "github.com",
    "ebay.com", "dropbox.com", "spotify.com",
    "chase.com", "wellsfargo.com", "bankofamerica.com",
    "steampowered.com", "store.steampowered.com",
    "coinbase.com", "binance.com",
    "runescape.com", "discord.com", "twitch.tv",
}


def normalize_url(url: str) -> dict:
    """
    Normalize a URL and return both the normalized form
    and metadata about what was stripped.

    Returns:
        {
            "normalized": bare URL without www/scheme noise,
            "original": original URL,
            "had_www": bool,
            "had_https": bool,
            "had_http": bool,
            "had_no_scheme": bool,
            "root_domain": just the domain (e.g. "google.com"),
            "is_exact_brand": whether it exactly matches a known legit domain
        }
    """
    url = str(url).strip()
    original = url

    # Detect scheme
    had_https    = url.lower().startswith("https://")
    had_http     = url.lower().startswith("http://") and not had_https
    had_no_scheme = not (had_https or had_http)

    # Add scheme for parsing if missing
    parse_url = url if (had_https or had_http) else "http://" + url

    try:
        parsed = urlparse(parse_url)
        hostname = (parsed.hostname or "").lower()
        path     = parsed.path or ""
        query    = parsed.query or ""
        fragment = parsed.fragment or ""

        # Detect and strip www.
        had_www = hostname.startswith("www.")
        clean_hostname = hostname[4:] if had_www else hostname

        # Root domain (last 2 parts)
        parts = clean_hostname.split(".")
        root_domain = ".".join(parts[-2:]) if len(parts) >= 2 else clean_hostname

        # Rebuild normalized URL (no scheme, no www.)
        normalized = clean_hostname
        if path and path != "/":
            normalized += path
        if query:
            normalized += "?" + query
        if fragment:
            normalized += "#" + fragment

        # Check exact brand match
        is_exact_brand = root_domain in LEGITIMATE_EXACT_DOMAINS

    except Exception:
        normalized    = url
        clean_hostname = ""
        root_domain    = ""
        had_www        = False
        is_exact_brand = False

    return {
        "normalized":    normalized,
        "original":      original,
        "had_www":       had_www,
        "had_https":     had_https,
        "had_http":      had_http,
        "had_no_scheme": had_no_scheme,
        "root_domain":   root_domain,
        "is_exact_brand": is_exact_brand,
    }


def get_normalization_features(url: str) -> dict:
    """
    Returns features derived from URL normalization.
    These REPLACE the raw scheme features and fix the bias.
    """
    meta = normalize_url(url)

    return {
        # Scheme features (same as before but cleaner)
        "had_https":          int(meta["had_https"]),
        "had_http":           int(meta["had_http"]),
        "had_no_scheme":      int(meta["had_no_scheme"]),

        # www fix — www on its own is NOT a phishing signal
        # but www. on a phishing domain often means deliberate mimicry
        "had_www":            int(meta["had_www"]),

        # KEY FIX: exact brand match means it IS the real site
        # This prevents google.com from being flagged by brand_confusion_score
        "is_exact_brand":     int(meta["is_exact_brand"]),
    }


# ─────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        ("https://google.com",              "LEGIT"),
        ("https://www.google.com",          "LEGIT"),
        ("http://www.paypal.com",           "LEGIT"),
        ("paypal.com",                      "LEGIT - dataset format"),
        ("www.facebook.com/login",          "LEGIT"),
        ("http://paypa1.com/login",         "PHISHING"),
        ("https://secure-paypal.tk",        "PHISHING"),
        ("http://www.paypal.com.evil.tk",   "PHISHING"),
        ("google.com.phish-login.xyz",      "PHISHING"),
    ]

    print("=" * 75)
    print("URL NORMALIZER TEST")
    print("=" * 75)
    print(f"{'URL':<42} {'HTTPS':>6} {'HTTP':>5} {'WWW':>5} {'EXACT_BRAND':>12}  LABEL")
    print("-" * 75)

    for url, label in test_cases:
        f = get_normalization_features(url)
        meta = normalize_url(url)
        print(
            f"{url:<42} "
            f"{f['had_https']:>6} "
            f"{f['had_http']:>5} "
            f"{f['had_www']:>5} "
            f"{f['is_exact_brand']:>12}  "
            f"{label}"
        )
        print(f"  → normalized: {meta['normalized']}")
