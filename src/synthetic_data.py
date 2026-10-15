"""Generate a SYNTHETIC learner-course interaction dataset.

WHY THIS EXISTS: collaborative filtering needs a user x item interaction
matrix. The Coursera catalogue contains no user data, and no public dataset
with genuine learner IDs that maps onto this catalogue was available
(the public Coursera *reviews* dataset identifies reviewers only by display
names such as "By Robert S", which are not unique users). Rather than
pretend, we simulate learners so the CF algorithm can be demonstrated.

WHAT IT IS NOT: real learner behaviour. Every number produced from this file
reflects the assumptions of the generator below, and is labelled synthetic
everywhere it is shown.

Generator (seeded, reproducible):
1. Each synthetic learner gets a persona = (career goal from the manual
   knowledge file, starting level).
2. Each course receives a random latent "appeal" (log-normal) so that some
   courses are taken far more often than others, as on real platforms.
   This appeal is random and is NOT derived from ratings.
3. A learner picks courses with probability proportional to
       appeal x subdomain affinity x level affinity
   where subdomain affinity favours the persona's related Coursera subdomains,
   plus a fixed share of uniformly random "exploration" picks.
4. Picks are ordered from easier to harder to mimic progression; the order is
   stored so the evaluation can hold out each learner's *latest* courses.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import SYNTHETIC_INTERACTIONS_PATH
from src.knowledge_base import SkillKnowledge

RANDOM_SEED = 42
NUMBER_OF_LEARNERS = 1200
MIN_COURSES_PER_LEARNER = 4
MEAN_EXTRA_COURSES = 6            # Poisson mean added to the minimum
EXPLORATION_SHARE = 0.15          # share of picks drawn uniformly at random
APPEAL_SIGMA = 1.0                # spread of the log-normal course appeal

RELATED_SUBDOMAIN_AFFINITY = 1.0
SAME_DOMAIN_AFFINITY = 0.15
OTHER_AFFINITY = 0.01
LEVEL_AFFINITY = {0: 1.0, 1: 0.5, 2: 0.2}   # by |course level - learner level|
UNKNOWN_LEVEL_AFFINITY = 0.5


def generate_interactions(
    courses: pd.DataFrame,
    knowledge: SkillKnowledge,
    n_learners: int = NUMBER_OF_LEARNERS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Return a long table of synthetic (learner, course) completions."""
    rng = np.random.default_rng(seed)
    course_ids = courses["course_id"].to_numpy()
    subdomains = courses["subdomain"].to_numpy()
    domains = courses["domain"].to_numpy()
    difficulty = courses["difficulty_rank"].to_numpy()
    appeal = rng.lognormal(mean=0.0, sigma=APPEAL_SIGMA, size=len(courses))

    careers = list(knowledge.careers)
    subdomain_to_domain = dict(zip(subdomains, domains))
    rows = []
    for learner_number in range(n_learners):
        career = careers[rng.integers(len(careers))]
        level = int(rng.integers(1, 4))
        related = set(knowledge.careers[career]["related_subdomains"])
        related_domains = {subdomain_to_domain.get(s) for s in related}

        subdomain_affinity = np.where(
            np.isin(subdomains, list(related)), RELATED_SUBDOMAIN_AFFINITY,
            np.where(np.isin(domains, list(related_domains)), SAME_DOMAIN_AFFINITY, OTHER_AFFINITY),
        )
        level_affinity = np.full(len(courses), UNKNOWN_LEVEL_AFFINITY)
        known_level = ~np.isnan(difficulty)
        level_gap = np.abs(difficulty[known_level] - level).astype(int)
        level_affinity[known_level] = [LEVEL_AFFINITY[gap] for gap in level_gap]
        weights = appeal * subdomain_affinity * level_affinity
        probabilities = weights / weights.sum()

        n_courses = MIN_COURSES_PER_LEARNER + rng.poisson(MEAN_EXTRA_COURSES)
        n_random = rng.binomial(n_courses, EXPLORATION_SHARE)
        chosen = list(rng.choice(len(courses), size=n_courses - n_random, replace=False, p=probabilities))
        remaining = np.setdiff1d(np.arange(len(courses)), chosen)
        chosen += list(rng.choice(remaining, size=n_random, replace=False))

        # Order easier -> harder (unknown difficulty treated as Intermediate),
        # random tie-break, to give each learner a plausible progression.
        order = sorted(chosen, key=lambda i: (np.nan_to_num(difficulty[i], nan=2.0), rng.random()))
        for step, course_index in enumerate(order):
            rows.append({
                "learner_id": f"synthetic_{learner_number:04d}",
                "course_id": course_ids[course_index],
                "sequence": step,
                "persona_career": career,
                "persona_level": level,
                "is_synthetic": True,
            })
    return pd.DataFrame(rows)


def save_interactions(interactions: pd.DataFrame, path: Path = SYNTHETIC_INTERACTIONS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(path, index=False)


def load_interactions(path: Path = SYNTHETIC_INTERACTIONS_PATH) -> pd.DataFrame | None:
    """Load the synthetic interactions, or ``None`` if they have not been generated."""
    if not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    if "is_synthetic" not in frame.columns or not frame["is_synthetic"].all():
        raise ValueError(f"{path} is not labelled as synthetic data; refusing to load it.")
    return frame


if __name__ == "__main__":
    from src.knowledge_base import load_knowledge
    from src.preprocessing import load_processed

    generated = generate_interactions(load_processed(), load_knowledge())
    save_interactions(generated)
    print(f"Wrote {len(generated)} SYNTHETIC interactions for "
          f"{generated['learner_id'].nunique()} synthetic learners to {SYNTHETIC_INTERACTIONS_PATH}")
