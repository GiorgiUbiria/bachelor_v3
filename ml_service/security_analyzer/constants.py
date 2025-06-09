import re

XSS_PATTERNS = [
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'javascript:', re.IGNORECASE),
    re.compile(r'<img[^>]*onerror[^>]*>', re.IGNORECASE),
    re.compile(r'<svg[^>]*onload[^>]*>', re.IGNORECASE),
    re.compile(r'alert\s*\(', re.IGNORECASE),
    re.compile(r'document\.cookie', re.IGNORECASE)
]

SQLI_PATTERNS = [
    re.compile(r"'\s*(or|and)\s*'", re.IGNORECASE),
    re.compile(r'union\s+select', re.IGNORECASE),
    re.compile(r'drop\s+table', re.IGNORECASE),
    re.compile(r'--\s*$', re.MULTILINE),
    re.compile(r"'\s*;\s*drop", re.IGNORECASE),
    re.compile(r'1\s*=\s*1', re.IGNORECASE)
]

CSRF_INDICATORS = [
    'missing_csrf_token',
    'referer_mismatch',
    'suspicious_referer'
]

ATTACK_TYPES = ['xss', 'sqli', 'csrf', 'benign'] 