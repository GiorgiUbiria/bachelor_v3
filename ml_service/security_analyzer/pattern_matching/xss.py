from ..constants import XSS_PATTERNS
from typing import List, Tuple

class XSSDetector:
    def __init__(self):
        self.patterns = XSS_PATTERNS
    
    def detect(self, text: str) -> Tuple[bool, List[str]]:
        matches = []
        for pattern in self.patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        return len(matches) > 0, matches 