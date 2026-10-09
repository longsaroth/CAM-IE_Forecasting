"""Khmer date parsing utilities for CAM-IE registration reports."""
import re
from datetime import date
from typing import Optional

KHMER_DIGITS = {
    '០': '0', '១': '1', '២': '2', '៣': '3', '៤': '4',
    '៥': '5', '៦': '6', '៧': '7', '៨': '8', '៩': '9'
}

KHMER_MONTHS = {
    'មករា': 1,
    'កុម្ភៈ': 2, 'កុម្ភះ': 2,
    'មីនា': 3,
    'មេសា': 4,
    'ឧសភា': 5,
    'មិថុនា': 6,
    'កក្កដា': 7,
    'សីហា': 8,
    'កញ្ញា': 9,
    'តុលា': 10,
    'វិច្ឆិកា': 11,
    'ធ្នូ': 12
}

def khmer_to_arabic_digits(text: str) -> str:
    """Converts Khmer numerals (០-៩) to Arabic numerals (0-9)."""
    return "".join(KHMER_DIGITS.get(ch, ch) for ch in text)

def parse_khmer_date(date_str: str) -> Optional[str]:
    """
    Parses strings like '១ កញ្ញា ២០២៦' into '2026-09-01'.
    Falls back gracefully if the input is already ISO formatted.
    """
    if not date_str:
        return None
    
    # Clean zero-width spaces and normalize whitespace
    cleaned = re.sub(r'[\u200b\u00a0\s]+', ' ', date_str).strip()
    
    # Check if already ISO format YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', cleaned):
        return cleaned

    # Expected pattern: {Khmer/Arabic Day} {Khmer Month} {Khmer/Arabic Year}
    parts = cleaned.split()
    if len(parts) >= 3:
        day_str = khmer_to_arabic_digits(parts[0])
        month_name = parts[1]
        year_str = khmer_to_arabic_digits(parts[2])
        
        try:
            day = int(day_str)
            month = KHMER_MONTHS.get(month_name)
            year = int(year_str)
            if month and 1 <= day <= 31 and 2000 <= year <= 2100:
                return f"{year:04d}-{month:02d}-{day:02d}"
        except (ValueError, TypeError):
            pass
            
    return None
