from ..constants import SQLI_PATTERNS
from typing import List, Tuple

class SQLIDetector:
    def __init__(self):
        self.patterns = SQLI_PATTERNS
    
    def detect(self, text: str) -> Tuple[bool, List[str]]:
        matches = []
        for pattern in self.patterns:
            if pattern.search(text):
                matches.append(pattern.pattern)
        
        return len(matches) > 0, matches 