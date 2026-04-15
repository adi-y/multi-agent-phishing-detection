import re
import math
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs

from agents.url_agent.ngram_features import extract_ngram_features
from agents.url_agent.url_resolver import resolve_url, is_shortened
from agents.url_agent.whois_features import extract_whois_features
from agents.url_agent.url_normalizer import normalize_url, get_normalization_features

# --- Suspicious signals ---
SUSPICIOUS_WORDS = [
    "login", "secure", "verify", "update", "bank", "account", "signin",
    "webscr", "ebayisapi", "confirm", "password", "credential", "submit",
    "paypal", "free", "lucky", "prize", "winner", "click", "redirect",
    "recovery", "restore", "unlock", "reset", "validate"
]

TRUSTED_BRANDS = [
    "paypal", "google", "facebook", "apple", "microsoft", "amazon",
    "netflix", "instagram", "twitter", "linkedin", "ebay", "chase",
    "wellsfargo", "bankofamerica", "citibank", "steam", "runescape"
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".info", ".biz", ".online", ".site", ".website", ".store",
    ".live", ".stream", ".download", ".click", ".link"
]

FREE_HOSTING = [
    "000webhostapp", "weebly", "wix", "wordpress.com", "blogspot",
    "tripod", "angelfire", "godaddysites", "joomla", "000webhost",
    "htmldrop", "x10host", "biz.nf", "altervista"
]


# ─────────────────────────────────────────────
# Helper functions — all accept bare URLs
# ─────────────────────────────────────────────

def has_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0


def get_entropy(s):
    if not s:
        return 0
    prob = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob if p > 0)


def count_suspicious_words(url):
    url_lower = url.lower()
    return sum(word in url_lower for word in SUSPICIOUS_WORDS)


def has_brand_in_subdomain(url):
    """Brand name in subdomain but not as the real domain (e.g. paypal.evil.com)"""
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        subdomain_parts = parts[:-2] if len(parts) > 2 else []
        subdomain = ".".join(subdomain_parts).lower()
        return int(any(brand in subdomain for brand in TRUSTED_BRANDS))
    except Exception:
        return 0


def uses_free_hosting(url):
    url_lower = url.lower()
    return int(any(host in url_lower for host in FREE_HOSTING))


def count_digits_in_domain(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        return sum(c.isdigit() for c in hostname)
    except Exception:
        return 0


def has_suspicious_tld(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        return int(any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS))
    except Exception:
        return 0


def count_query_params(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        return len(parse_qs(parsed.query))
    except Exception:
        return 0


def get_path_depth(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        return parsed.path.count("/")
    except Exception:
        return 0


def has_double_slash_redirect(url):
    # Only check after the first 7 chars to skip protocol slashes
    return 1 if "//" in url[7:] else 0


def get_domain_length(url):
    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        root = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
        return len(root)
    except Exception:
        return 0


# ─────────────────────────────────────────────
# Main feature extraction
# ─────────────────────────────────────────────

def extract_features(url: str, live_whois: bool = False) -> dict:
    """
    Extract all 47 features from a URL.

    THE CRITICAL FIX:
        Structural features (counts, lengths, entropy) are extracted
        from the NORMALIZED URL (no scheme, no www.) — exactly matching
        how the training data was formatted.

        Scheme and www features are extracted from the ORIGINAL URL
        before normalization, so those signals are preserved.

    Args:
        url:        Raw URL string (with or without scheme)
        live_whois: If True, performs live WHOIS lookup (for real-time API).
                    If False, WHOIS features return -1 (fast, for batch/test).
    """
    url = str(url).strip()

    # ── Gap 1: Resolve shortened URLs first ──────────────────────────────
    was_shortened = is_shortened(url)
    if was_shortened:
        url = resolve_url(url)

    # ── Normalization: extract scheme/www BEFORE stripping ───────────────
    # norm_meta holds: normalized, had_www, had_https, had_http,
    #                  had_no_scheme, root_domain, is_exact_brand
    norm_meta  = normalize_url(url)
    norm_feats = get_normalization_features(url)

    # All structural features computed on bare URL (matches training format)
    # e.g. "https://www.google.com/search?q=1" → "google.com/search?q=1"
    bare = norm_meta["normalized"]

    # Add http:// for urlparse to work correctly on the bare URL
    bare_with_scheme = "http://" + bare if "://" not in bare else bare

    try:
        parsed      = urlparse(bare_with_scheme)
        hostname    = parsed.hostname or ""
        path        = parsed.path or ""
        query       = parsed.query or ""
        fragment    = parsed.fragment or ""
    except Exception:
        parsed   = None
        hostname = ""
        path     = ""
        query    = ""
        fragment = ""

    # ── Structural features (18) ─────────────────────────────────────────
    url_length     = len(bare)
    domain_len     = get_domain_length(bare)
    path_len       = len(path)
    query_len      = len(query)

    num_dots            = bare.count(".")
    num_hyphens         = bare.count("-")
    num_underscores     = bare.count("_")
    num_slashes         = bare.count("/")
    num_question_marks  = bare.count("?")
    num_ampersands      = bare.count("&")
    num_equals          = bare.count("=")
    num_at_symbols      = bare.count("@")
    num_percent         = bare.count("%")
    num_digits_in_url   = sum(c.isdigit() for c in bare)
    num_digits_in_domain = count_digits_in_domain(bare)

    # Subdomains: parts of hostname minus the root 2 (e.g. a.b.google.com → 2 subdomains)
    parts = hostname.split(".") if hostname else []
    num_subdomains = max(0, len(parts) - 2)

    depth           = get_path_depth(bare)
    num_query_params = count_query_params(bare)

    # ── Boolean signals (13) ────────────────────────────────────────────
    ip_flag             = has_ip(bare)
    at_flag             = 1 if "@" in bare else 0
    double_slash_flag   = has_double_slash_redirect(bare)
    port_flag           = 1 if parsed and parsed.port else 0
    fragment_flag       = 1 if fragment else 0
    suspicious_tld_flag = has_suspicious_tld(bare)
    free_hosting_flag   = uses_free_hosting(bare)
    brand_subdomain_flag = has_brand_in_subdomain(bare)
    shortened_flag      = int(was_shortened)
    exact_brand_flag    = norm_feats["is_exact_brand"]

    # scheme_mismatch: brand domain on plain HTTP (from original URL)
    from agents.url_agent.scheme_features import get_scheme_features
    scheme_feats   = get_scheme_features(url)          # uses original url
    scheme_mismatch = scheme_feats["scheme_mismatch"]

    had_www_flag        = norm_feats["had_www"]
    misleading_prefix   = 0
    try:
        misleading_prefixes = ["secure-", "login-", "verify-", "update-",
                                "account-", "confirm-", "banking-", "safe-"]
        misleading_prefix = int(any(hostname.startswith(p) for p in misleading_prefixes))
    except Exception:
        pass

    # ── Scheme features (3) — from ORIGINAL url ─────────────────────────
    had_https    = norm_feats["had_https"]
    had_http     = norm_feats["had_http"]
    had_no_scheme = norm_feats["had_no_scheme"]

    # ── Entropy features (3) ─────────────────────────────────────────────
    url_entropy    = get_entropy(bare)
    domain_entropy = get_entropy(hostname)
    path_entropy   = get_entropy(path)

    # ── Ratio features (2) ───────────────────────────────────────────────
    digit_ratio       = num_digits_in_url / max(url_length, 1)
    special_chars     = sum(1 for c in bare if not c.isalnum() and c not in "-._~/")
    special_char_ratio = special_chars / max(url_length, 1)

    # ── N-gram / obfuscation features (8) — Gap 2 ───────────────────────
    # Pass bare URL so ngram functions see the same domain the model trained on
    ngram_feats = extract_ngram_features(bare)

    # ── WHOIS features (7) — Gap 5 ───────────────────────────────────────
    whois_feats = extract_whois_features(url, live_lookup=live_whois)

    # ── Assemble final feature dict ──────────────────────────────────────
    features = {
        # Structural (18)
        "url_length":            url_length,
        "domain_length":         domain_len,
        "path_length":           path_len,
        "query_length":          query_len,
        "num_dots":              num_dots,
        "num_hyphens":           num_hyphens,
        "num_underscores":       num_underscores,
        "num_slashes":           num_slashes,
        "num_question_marks":    num_question_marks,
        "num_ampersands":        num_ampersands,
        "num_equals":            num_equals,
        "num_at_symbols":        num_at_symbols,
        "num_percent":           num_percent,
        "num_digits_in_url":     num_digits_in_url,
        "num_digits_in_domain":  num_digits_in_domain,
        "num_subdomains":        num_subdomains,
        "path_depth":            depth,
        "num_query_params":      num_query_params,

        # Boolean signals (13)
        "has_ip":                ip_flag,
        "has_at_symbol":         at_flag,
        "has_double_slash":      double_slash_flag,
        "has_port":              port_flag,
        "has_fragment":          fragment_flag,
        "has_suspicious_tld":    suspicious_tld_flag,
        "uses_free_hosting":     free_hosting_flag,
        "has_brand_in_subdomain": brand_subdomain_flag,
        "is_shortened":          shortened_flag,
        "is_exact_brand":        exact_brand_flag,
        "scheme_mismatch":       scheme_mismatch,
        "had_www":               had_www_flag,
        "has_misleading_prefix": misleading_prefix,

        # Scheme (3)
        "had_https":             had_https,
        "had_http":              had_http,
        "had_no_scheme":         had_no_scheme,

        # Entropy (3)
        "url_entropy":           url_entropy,
        "domain_entropy":        domain_entropy,
        "path_entropy":          path_entropy,

        # Ratio (2)
        "digit_ratio":           digit_ratio,
        "special_char_ratio":    special_char_ratio,

        # N-gram / obfuscation (8)
        "brand_ngram_similarity":   ngram_feats["brand_ngram_similarity"],
        "brand_trigram_similarity":  ngram_feats["brand_trigram_similarity"],
        "min_brand_edit_distance":   ngram_feats["min_brand_edit_distance"],
        "is_typosquat":              ngram_feats["is_typosquat"],
        "has_homograph_chars":       ngram_feats["has_homograph_chars"],
        "brand_confusion_score":     ngram_feats["brand_confusion_score"],
        "repeated_char_count":       ngram_feats["repeated_char_count"],
        "suspicious_word_count":     count_suspicious_words(bare),

        # WHOIS (7)
        "domain_age_days":           whois_feats["domain_age_days"],
        "days_until_expiry":         whois_feats["days_until_expiry"],
        "registration_period":       whois_feats["registration_period"],
        "is_new_domain":             whois_feats["is_new_domain"],
        "is_very_new_domain":        whois_feats["is_very_new_domain"],
        "is_short_registration":     whois_feats["is_short_registration"],
        "whois_lookup_failed":       whois_feats["whois_lookup_failed"],
    }

    return features