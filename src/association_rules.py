"""Association rule mining over skills that are TAUGHT TOGETHER in courses.

Transactions: one per course; items: the skills from the knowledge vocabulary
that the course directly addresses (tag/title evidence).

This mines the structure of the *catalogue*, e.g. "courses that teach Deep
Learning usually also teach Machine Learning". It is NOT learner behaviour
(there is no real enrolment data), and the rules are never used to rank
courses; they are shown on the skill-gap page as related skills.

Measures for a rule A -> B over N transactions:
    support(A -> B)    = count(A and B) / N
    confidence(A -> B) = support(A and B) / support(A)
    lift(A -> B)       = confidence(A -> B) / support(B)
Lift > 1 means A and B co-occur more often than if they were independent.
"""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from src.knowledge_base import ADDRESSES_THRESHOLD

MIN_SUPPORT = 0.01        # the itemset appears in at least 1% of courses (~34)
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.0


def mine_skill_rules(skill_matrix: pd.DataFrame, min_support: float = MIN_SUPPORT,
                     min_confidence: float = MIN_CONFIDENCE) -> pd.DataFrame:
    """Apriori on the course x skill boolean matrix; returns rules sorted by lift."""
    transactions = skill_matrix >= ADDRESSES_THRESHOLD
    transactions = transactions[transactions.any(axis=1)]
    if transactions.empty:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    itemsets = apriori(transactions, min_support=min_support, use_colnames=True, max_len=2)
    if itemsets.empty or itemsets["itemsets"].map(len).max() < 2:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    rules = rules[rules["lift"] > MIN_LIFT]
    table = pd.DataFrame({
        "antecedent": rules["antecedents"].map(lambda s: ", ".join(sorted(s))),
        "consequent": rules["consequents"].map(lambda s: ", ".join(sorted(s))),
        "support": rules["support"].round(4),
        "confidence": rules["confidence"].round(3),
        "lift": rules["lift"].round(2),
    })
    return table.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)


def related_skills(rules: pd.DataFrame, skill: str, top: int = 3) -> pd.DataFrame:
    """Rules whose antecedent is ``skill``."""
    return rules[rules["antecedent"] == skill].head(top)
