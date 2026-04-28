from core.loader import load_dataset
from core.reasoner import run_analysis
data = load_dataset(open('askfirst_synthetic_dataset.json'))
for u in data['users']:
    r = run_analysis(u, use_llm=False)
    print(f"User {u['user_id']}:")
    for p in r.patterns:
        print(f"  - {p.pattern_title} ({p.pattern_type}, {p.confidence_label}, {p.confidence_score})")
