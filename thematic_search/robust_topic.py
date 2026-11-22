"""
Robust Topic Estimator.

Implements the RobustTopic operation from the theory (Subsection 2.5 and Section 3.4):

    RobustTopic(f) = argmax_{φ ∈ 𝒢_T} { S(f,φ) - λ · (log(1+|f|)/|f|) · r(φ) }

where S(f,φ) = c(f,φ) · log(c(f,φ)/p(φ)) is a surprise score inspired by
KL-divergence, and r(φ) is the minimum description rank (number of ∧ and ∨
operators) of the formula.

The argmax is computed using the greedy algorithm from Section 3.4, which
alternates disjunction and conjunction steps over vertices of T.
"""
import numpy as np
from .topic_formula import TopicFormula, simplify_formula


def _topic_name_map(db) -> dict | None:
    """Build a vertex index -> display name mapping from db.topic_df."""
    if db.topic_df is None or "name" not in db.topic_df.columns:
        return None
    return dict(zip(db.topic_df.index, db.topic_df["name"]))


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _surprise(f: np.ndarray, clust_phi: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the surprise score S(f, φ).

    Parameters
    ----------
    f : np.ndarray, shape (n_docs,)
        Fuzzy query set, values in [0, 1].
    clust_phi : np.ndarray, shape (n_docs,)
        Inclusion strength vector Clust(φ), values in [0, 1].
    eps : float
        Small value to avoid log(0) and division by zero.

    Returns
    -------
    float
        The surprise score S(f, φ) = c · log(c / p).
    """
    f_size = f.sum()
    if f_size < eps:
        return 0.0

    # Coverage: c(f, φ) = (1/|f|) Σ_x min{Clust(φ)(x), f(x)}
    coverage = np.minimum(clust_phi, f).sum() / f_size
    if coverage < eps:
        return 0.0

    # Prior: p(φ) = (1/|X|) Σ_x Clust(φ)(x)
    prior = clust_phi.mean()
    if prior < eps:
        return 0.0

    return float(coverage * np.log(coverage / prior))


def _objective(
    f: np.ndarray,
    clust_phi: np.ndarray,
    rank: int,
    lambda_: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute the objective function L(φ) = S(f,φ) - λ · (log(1+|f|)/|f|) · r(φ).

    Parameters
    ----------
    f : np.ndarray, shape (n_docs,)
        Fuzzy query set.
    clust_phi : np.ndarray, shape (n_docs,)
        Inclusion strength vector Clust(φ).
    rank : int
        The description rank r(φ).
    lambda_ : float
        Complexity penalty strength.
    eps : float
        Small value to avoid numerical issues.

    Returns
    -------
    float
    """
    s = _surprise(f, clust_phi, eps=eps)

    f_size = f.sum()
    if f_size < eps:
        return s

    penalty = lambda_ * (np.log(1.0 + f_size) / f_size) * rank
    return s - penalty


# ---------------------------------------------------------------------------
# Formula evaluation from cached vertex vectors
# ---------------------------------------------------------------------------

def _eval_formula_dnf(terms: list[set[int]], all_strengths: np.ndarray) -> np.ndarray:
    """
    Evaluate Clust(φ) for a formula φ in DNF, using precomputed vertex vectors.

    For a DNF formula φ = (v₁∧v₂) ∨ (v₃∧v₄) ∨ ...:
        Clust(φ) = max_i( min_{v ∈ term_i}( Clust(v) ) )

    Parameters
    ----------
    terms : list[set[int]]
        The DNF terms.
    all_strengths : np.ndarray, shape (n_docs, n_topics)
        Precomputed float inclusion strength matrix.

    Returns
    -------
    np.ndarray, shape (n_docs,)
    """
    if not terms:
        return np.zeros(all_strengths.shape[0])

    term_clusts = []
    for term in terms:
        vertices = list(term)
        tc = all_strengths[:, vertices[0]].copy()
        for v in vertices[1:]:
            np.minimum(tc, all_strengths[:, v], out=tc)
        term_clusts.append(tc)

    if len(term_clusts) == 1:
        return term_clusts[0]

    return np.maximum.reduce(term_clusts)


# ---------------------------------------------------------------------------
# Greedy RobustTopic algorithm
# ---------------------------------------------------------------------------

def robust_topic(
    db,
    f: np.ndarray,
    lambda_: float = 1.0,
    candidates_per_step: int = 3,
) -> TopicFormula:
    """
    Compute RobustTopic(f) using beam search over the greedy algorithm from Section 3.4.

    The algorithm alternates disjunction and conjunction steps, maintaining the top
    `candidates_per_step` candidates at each stage. This helps escape local optima
    by exploring multiple promising paths simultaneously.

    Parameters
    ----------
    db : TopicDatabase
        The topic database.
    f : np.ndarray, shape (n_docs,)
        A fuzzy subset of the documents, values in [0, 1].
        Must be non-zero (i.e., f.sum() > 0).
    lambda_ : float, optional (default=1.0)
        Strength of the complexity penalty. Must be > 0.
    candidates_per_step : int, optional (default=3)
        Number of candidates to maintain in the beam search. Higher values
        explore more paths but increase computation time.

    Returns
    -------
    TopicFormula
        The robust topic formula φ ∈ 𝒢_T.

    Raises
    ------
    ValueError
        If f is all zeros.
    """
    if f.sum() < 1e-12:
        raise ValueError(
            "robust_topic requires a non-zero fuzzy set f (f.sum() > 0)."
        )

    tree = db.soft_cluster_tree
    all_strengths = db._all_strengths  # (n_docs, n_topics), float in [0, 1]
    vertices = list(tree.idx_to_loc.keys())
    adj = tree.adjacency_closure
    topic_names = _topic_name_map(db)

    # --- Step 1: Initialize with top candidates_per_step vertices ---
    # Score all vertices
    init_candidates = []
    for v in vertices:
        clust_v = all_strengths[:, v]
        L = _objective(f, clust_v, rank=0, lambda_=lambda_)
        init_candidates.append((L, v, clust_v.copy()))
    
    # Sort by score descending and take top k
    init_candidates.sort(key=lambda x: -x[0])
    init_candidates = init_candidates[:candidates_per_step]
    
    # Convert to formula objects
    # Each candidate: (score, formula, clust)
    candidates = []
    for L, v, clust_v in init_candidates:
        formula = TopicFormula.from_vertex(v, topic_names=topic_names)
        candidates.append((L, formula, clust_v))

    # --- Step 2: Beam search loop ---
    while True:
        new_candidates = []
        
        # For each candidate, try disjunction with each vertex
        for L, formula, clust_phi in candidates:
            # Disjunction: try φ ∨ v for all v
            for v in vertices:
                # Clust(φ ∨ v) = max(Clust(φ), Clust(v))
                new_clust = np.maximum(clust_phi, all_strengths[:, v])
                new_terms = formula.terms + [{v}]
                new_rank = sum(len(t) - 1 for t in new_terms) + len(new_terms) - 1
                new_L = _objective(f, new_clust, rank=new_rank, lambda_=lambda_)
                
                # Only accept if strictly better
                if new_L > L:
                    new_formula = TopicFormula(new_terms, topic_names=topic_names)
                    new_formula = simplify_formula(new_formula, adj)
                    new_clust_eval = _eval_formula_dnf(new_formula.terms, all_strengths)
                    new_candidates.append((new_L, new_formula, new_clust_eval))
            
            # Conjunction: try φ ∧ v for all v
            for v in vertices:
                # ψ ∧ v: distribute v into all terms
                new_terms = [t | {v} for t in formula.terms]
                new_clust = _eval_formula_dnf(new_terms, all_strengths)
                new_rank = sum(len(t) - 1 for t in new_terms) + len(new_terms) - 1
                new_L = _objective(f, new_clust, rank=new_rank, lambda_=lambda_)
                
                # Only accept if strictly better
                if new_L > L:
                    new_formula = TopicFormula(new_terms, topic_names=topic_names)
                    new_formula = simplify_formula(new_formula, adj)
                    new_clust_eval = _eval_formula_dnf(new_formula.terms, all_strengths)
                    new_candidates.append((new_L, new_formula, new_clust_eval))
        
        # If no improvements found, terminate
        if not new_candidates:
            break
        
        # Retain current candidates (local optima) so they compete
        # with newly improved candidates in the next round
        new_candidates.extend(candidates)
        
        # Sort by score and keep top candidates_per_step
        new_candidates.sort(key=lambda x: -x[0])
        candidates = new_candidates[:candidates_per_step]
    
    # Return the best candidate
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]
