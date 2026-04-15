import re
import math
from urllib.parse import urlparse, parse_qs

from agents.url_agent.ngram_features import extract_ngram_features
from agents.url_agent.url_resolver import resolve_url, is_shortened
from agents.url_agent.whois_features import extract_whois_features
from agents.url_agent.url_normalizer import normalize_url, get_normalization_features
from agents.url_agent.scheme_features import get_scheme_features

# ── Suspicious signals ────────────────────────────────────────────────────────
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

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".info", ".biz", ".online", ".site", ".website", ".store",
    ".live", ".stream", ".download", ".click", ".link"
}

FREE_HOSTING = {
    "000webhostapp", "weebly", "wix", "wordpress.com", "blogspot",
    "tripod", "angelfire", "godaddysites", "joomla", "000webhost",
    "htmldrop", "x10host", "biz.nf", "altervista"
}

MISLEADING_PREFIXES = (
    "secure-", "login-", "verify-", "update-",
    "account-", "confirm-", "banking-", "safe-"
)

SUSPICIOUS_WORDS_SET = set(SUSPICIOUS_WORDS)


# ── Helper functions ──────────────────────────────────────────────────────────

def has_ip(url):
    return 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0


def get_entropy(s):
    if not s:
        return 0.0
    length = len(s)
    return -sum(
        (cnt / length) * math.log2(cnt / length)
        for cnt in (s.count(c) for c in set(s))
        if cnt > 0
    )


def count_suspicious_words(url):
    url_lower = url.lower()
    return sum(1 for w in SUSPICIOUS_WORDS_SET if w in url_lower)


def has_brand_in_subdomain(hostname):
    """Pass already-parsed hostname for speed."""
    parts = hostname.split(".")
    if len(parts) <= 2:
        return 0
    subdomain = ".".join(parts[:-2]).lower()
    return int(any(brand in subdomain for brand in TRUSTED_BRANDS))


def uses_free_hosting(url_lower):
    return int(any(host in url_lower for host in FREE_HOSTING))


def count_digits_in_domain(hostname):
    return sum(c.isdigit() for c in hostname)


def has_suspicious_tld(hostname):
    return int(any(hostname.endswith(tld) for tld in SUSPICIOUS_TLDS))


def count_query_params(query):
    return len(parse_qs(query)) if query else 0


def calc_path_depth(path):
    return path.count("/")


def has_double_slash_redirect(url):
    check = url[7:] if len(url) > 7 else url
    return 1 if "//" in check else 0


def calc_domain_length(hostname):
    parts = hostname.split(".")
    root = ".".join(parts[-2:]) if len(parts) >= 2 else hostname
    return len(root)


# ── Public entry point ────────────────────────────────────────────────────────

def extract_features(url: str,
                     live_whois: bool = False,
                     skip_resolve: bool = False) -> dict:
    """
    Extract all features from a URL.

    Args:
        url:          Raw URL string (with or without scheme)
        live_whois:   If True, performs live WHOIS lookup.
                      Default False — fast for training and testing.
        skip_resolve: If True, skips URL shortener resolution entirely.
                      Always set True during training for speed.
                      Default False — resolution runs in real-time API.
    """
    try:
        return _extract(url, live_whois, skip_resolve)
    except Exception as e:
        import traceback
        print(f"[ERROR] extract_features failed for '{url}': {e}")
        traceback.print_exc()
        return None


def _extract(url: str, live_whois: bool, skip_resolve: bool) -> dict:
    url = str(url).strip()

    # ── Gap 1: resolve shortened URLs (skip during training) ──────────────
    was_shortened = False
    if not skip_resolve and is_shortened(url):
        was_shortened = True
        url = resolve_url(url)

    # ── Capture scheme/www from original BEFORE stripping ─────────────────
    norm_meta    = normalize_url(url)
    norm_feats   = get_normalization_features(url)
    scheme_feats = get_scheme_features(url)

    # ── All structural features use the bare URL ───────────────────────────
    # e.g. "https://www.google.com/path?q=1" → "google.com/path?q=1"
    bare = norm_meta["normalized"]
    bare_for_parse = "http://" + bare if "://" not in bare else bare

    try:
        parsed   = urlparse(bare_for_parse)
        hostname = parsed.hostname or ""
        path     = parsed.path or ""
        query    = parsed.query or ""
        fragment = parsed.fragment or ""
        port     = parsed.port
    except Exception:
        hostname = ""
        path     = ""
        query    = ""
        fragment = ""
        port     = None

    bare_lower = bare.lower()
    parts      = hostname.split(".") if hostname else []

    # ── Structural (18) ───────────────────────────────────────────────────
    url_length           = len(bare)
    domain_len           = calc_domain_length(hostname)
    path_len             = len(path)
    query_len            = len(query)
    num_dots             = bare.count(".")
    num_hyphens          = bare.count("-")
    num_underscores      = bare.count("_")
    num_slashes          = bare.count("/")
    num_question_marks   = bare.count("?")
    num_ampersands       = bare.count("&")
    num_equals           = bare.count("=")
    num_at_symbols       = bare.count("@")
    num_percent          = bare.count("%")
    num_digits_in_url    = sum(c.isdigit() for c in bare)
    num_digits_in_domain = count_digits_in_domain(hostname)
    num_subdomains       = max(0, len(parts) - 2)
    depth                = calc_path_depth(path)
    num_query_params     = count_query_params(query)

    # ── Boolean signals (13) ──────────────────────────────────────────────
    ip_flag              = has_ip(bare)
    at_flag              = 1 if "@" in bare else 0
    double_slash_flag    = has_double_slash_redirect(bare)
    port_flag            = 1 if port is not None else 0
    fragment_flag        = 1 if fragment else 0
    suspicious_tld_flag  = has_suspicious_tld(hostname)
    free_hosting_flag    = uses_free_hosting(bare_lower)
    brand_subdomain_flag = has_brand_in_subdomain(hostname)
    shortened_flag       = int(was_shortened)
    exact_brand_flag     = norm_feats["is_exact_brand"]
    scheme_mismatch      = scheme_feats["scheme_mismatch"]
    had_www_flag         = norm_feats["had_www"]
    misleading_prefix = int(hostname and any(hostname.startswith(p) for p in MISLEADING_PREFIXES))

    # ── Scheme (3) — from original url ────────────────────────────────────
    had_https     = norm_feats["had_https"]
    had_http      = norm_feats["had_http"]
    had_no_scheme = norm_feats["had_no_scheme"]

    # ── Entropy (3) ───────────────────────────────────────────────────────
    url_entropy    = get_entropy(bare)
    domain_entropy = get_entropy(hostname)
    path_entropy   = get_entropy(path)

    # ── Ratio (2) ─────────────────────────────────────────────────────────
    digit_ratio        = num_digits_in_url / max(url_length, 1)
    special_chars      = sum(1 for c in bare if not c.isalnum() and c not in "-._~/")
    special_char_ratio = special_chars / max(url_length, 1)

    # ── N-gram / obfuscation (8) — Gap 2 ──────────────────────────────────
    ngram_feats = extract_ngram_features(bare)

    # ── WHOIS (7) — Gap 5 ─────────────────────────────────────────────────
    whois_feats = {
    "domain_age_days": -1,
    "days_until_expiry": -1,
    "registration_period": -1,
    "is_new_domain": 0,
    "is_very_new_domain": 0,
    "is_short_registration": 0,
    "whois_lookup_failed": 1,
    }

    return {
        # Structural (18)
        "url_length":              url_length,
        "domain_length":           domain_len,
        "path_length":             path_len,
        "query_length":            query_len,
        "num_dots":                num_dots,
        "num_hyphens":             num_hyphens,
        "num_underscores":         num_underscores,
        "num_slashes":             num_slashes,
        "num_question_marks":      num_question_marks,
        "num_ampersands":          num_ampersands,
        "num_equals":              num_equals,
        "num_at_symbols":          num_at_symbols,
        "num_percent":             num_percent,
        "num_digits_in_url":       num_digits_in_url,
        "num_digits_in_domain":    num_digits_in_domain,
        "num_subdomains":          num_subdomains,
        "path_depth":              depth,
        "num_query_params":        num_query_params,
        # Boolean signals (13)
        "has_ip":                  ip_flag,
        "has_at_symbol":           at_flag,
        "has_double_slash":        double_slash_flag,
        "has_port":                port_flag,
        "has_fragment":            fragment_flag,
        "has_suspicious_tld":      suspicious_tld_flag,
        "uses_free_hosting":       free_hosting_flag,
        "has_brand_in_subdomain":  brand_subdomain_flag,
        "is_shortened":            shortened_flag,
        "is_exact_brand":          exact_brand_flag,
        "scheme_mismatch":         scheme_mismatch,
        "had_www":                 had_www_flag,
        "has_misleading_prefix":   misleading_prefix,
        # Scheme (3)
        "had_https":               had_https,
        "had_http":                had_http,
        "had_no_scheme":           had_no_scheme,
        # Entropy (3)
        "url_entropy":             url_entropy,
        "domain_entropy":          domain_entropy,
        "path_entropy":            path_entropy,
        # Ratio (2)
        "digit_ratio":             digit_ratio,
        "special_char_ratio":      special_char_ratio,
        # N-gram / obfuscation (8)
        "brand_ngram_similarity":  ngram_feats["brand_ngram_similarity"],
        "brand_trigram_similarity": ngram_feats["brand_trigram_similarity"],
        "min_brand_edit_distance":  ngram_feats["min_brand_edit_distance"],
        "is_typosquat":             ngram_feats["is_typosquat"],
        "has_homograph_chars":      ngram_feats["has_homograph_chars"],
        "brand_confusion_score":    ngram_feats["brand_confusion_score"],
        "repeated_char_count":      ngram_feats["repeated_char_count"],
        "suspicious_word_count":    count_suspicious_words(bare),
        # WHOIS (7)
        "domain_age_days":          whois_feats["domain_age_days"],
        "days_until_expiry":        whois_feats["days_until_expiry"],
        "registration_period":      whois_feats["registration_period"],
        "is_new_domain":            whois_feats["is_new_domain"],
        "is_very_new_domain":       whois_feats["is_very_new_domain"],
        "is_short_registration":    whois_feats["is_short_registration"],
        "whois_lookup_failed":      whois_feats["whois_lookup_failed"],
    }