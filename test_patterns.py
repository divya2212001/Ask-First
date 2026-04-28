"""
Test script to validate pattern detection against the synthetic dataset.
"""

from core.loader import load_dataset
from core.reasoner import run_analysis


def main():
    data = load_dataset("askfirst_synthetic_dataset.json")
    
    expected_patterns = {
        "USR001": [
            "late eating triggers stomach pain",
            "dehydration triggers headache",
        ],
        "USR002": [
            "calorie restriction triggers hair fall",
            "dairy triggers acne",
            "progressive symptom cascade from calorie restriction",
        ],
        "USR003": [
            "carbohydrate triggers fatigue",
            "sleep triggers cramps",
            "screen triggers fatigue",
            "screen triggers anxiety",
        ]
    }
    
    for user in data["users"]:
        uid = user["user_id"]
        print(f"\n{'='*60}")
        print(f"User {uid} - {user.get('name', 'Unknown')}")
        print(f"{'='*60}")
        
        result = run_analysis(user, use_llm=False)
        
        print(f"Patterns found: {len(result.patterns)}\n")
        
        for p in result.patterns:
            print(f"  • {p.pattern_title}")
            print(f"    Type: {p.pattern_type} | Confidence: {p.confidence_label} ({p.confidence_score})")
            print(f"    Justification: {p.confidence_justification}")
            print(f"    Reasoning: {p.reasoning_trace[:200]}...")
            if p.medical_latency_note:
                print(f"    Medical: {p.medical_latency_note}")
            print()
        
        # Check for expected patterns
        print("  --- Expected Pattern Check ---")
        for expected in expected_patterns.get(uid, []):
            found = any(expected.lower() in p.pattern_title.lower() for p in result.patterns)
            status = "✓" if found else "✗ MISSING"
            print(f"    {status} {expected}")
        
        print()


if __name__ == "__main__":
    main()

