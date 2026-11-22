"""
TopicFormula: Represents a geometric formula in 𝒢_T using disjunctive normal form (DNF).

A formula is represented as a disjunction of conjunctions:
    (v₁ ∧ v₂ ∧ ...) ∨ (v₃ ∧ v₄ ∧ ...) ∨ ...

where each term is a set of vertex indices (their conjunction).
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TopicFormula:
    """
    A geometric formula in 𝒢_T represented in disjunctive normal form.
    
    The formula is: (term₁) ∨ (term₂) ∨ ...
    where each term is a set of vertex indices representing their conjunction.
    
    Examples:
        TopicFormula([set()]) = False (empty formula)
        TopicFormula([{v}]) = v (single vertex)
        TopicFormula([{A, B}]) = A ∧ B (conjunction)
        TopicFormula([{A}, {B}]) = A ∨ B (disjunction)
        TopicFormula([{A, B}, {C, D}]) = (A ∧ B) ∨ (C ∧ D)
    
    Parameters
    ----------
    terms : list[set[int]]
        List of terms, where each term is a set of vertex indices.
        Each term represents a conjunction of those vertices.
    topic_names : dict[int, str], optional
        Mapping from vertex indices to display names. If provided, __repr__
        will use these names instead of vertex numbers.
    """
    terms: list[set[int]]
    topic_names: Optional[dict[int, str]] = None
    
    def __post_init__(self):
        """Ensure terms are sets."""
        self.terms = [set(t) for t in self.terms]
    
    @classmethod
    def from_vertex(cls, vertex_idx: int, topic_names: Optional[dict[int, str]] = None) -> 'TopicFormula':
        """Create a formula for a single vertex."""
        return cls([{vertex_idx}], topic_names=topic_names)
    
    @classmethod
    def from_vertices(cls, vertex_indices: set[int], topic_names: Optional[dict[int, str]] = None) -> 'TopicFormula':
        """Create a formula as conjunction of vertices."""
        return cls([vertex_indices], topic_names=topic_names)
    
    def __and__(self, other: 'TopicFormula') -> 'TopicFormula':
        """Conjunction: distribute ∧ over ∨."""
        # (A₁ ∨ A₂) ∧ (B₁ ∨ B₂) = (A₁∧B₁) ∨ (A₁∧B₂) ∨ (A₂∧B₁) ∨ (A₂∧B₂)
        new_terms = []
        for term_a in self.terms:
            for term_b in other.terms:
                new_terms.append(term_a | term_b)
        # Preserve topic_names from self if available, else from other
        names = self.topic_names if self.topic_names is not None else other.topic_names
        return TopicFormula(new_terms, topic_names=names)
    
    def __or__(self, other: 'TopicFormula') -> 'TopicFormula':
        """Disjunction: union of terms."""
        # Preserve topic_names from self if available, else from other
        names = self.topic_names if self.topic_names is not None else other.topic_names
        return TopicFormula(self.terms + other.terms, topic_names=names)
    
    @property
    def rank(self) -> int:
        """
        Compute the rank (minimum description length) of the formula.
        
        For a term with k vertices, rank is k-1 (k-1 ∧ operators).
        For a formula with n terms, rank is sum(term_ranks) + (n-1).
        """
        if not self.terms:
            return 0
        term_ranks = sum(len(t) - 1 for t in self.terms if t)
        disjunction_rank = len(self.terms) - 1
        return term_ranks + disjunction_rank
    
    def vertices(self) -> set[int]:
        """Return all vertices appearing in the formula."""
        return set().union(*self.terms) if self.terms else set()
    
    def is_atomic(self) -> bool:
        """Check if this is a single vertex (no operators)."""
        return len(self.terms) == 1 and len(self.terms[0]) == 1
    
    def to_index_expr(self) -> 'IndexExpr':
        """
        Convert this TopicFormula to an IndexExpr for evaluation.
        
        Returns
        -------
        IndexExpr
            Equivalent IndexExpr that can be evaluated by SoftClusterTree.
        """
        from .softclustertree import Cluster, IndexAnd, IndexOr
        
        if not self.terms:
            raise ValueError("Cannot convert empty formula to IndexExpr")
        
        # Convert each term to a conjunction
        term_exprs = []
        for term in self.terms:
            if not term:
                continue
            
            vertices = sorted(term)  # Sort for consistency
            expr = Cluster(vertices[0])
            for v in vertices[1:]:
                expr = IndexAnd(expr, Cluster(v))
            term_exprs.append(expr)
        
        # Combine terms with disjunction
        if not term_exprs:
            raise ValueError("Formula has no valid terms")
        
        result = term_exprs[0]
        for expr in term_exprs[1:]:
            result = IndexOr(result, expr)
        
        return result
    
    def _vertex_label(self, v: int) -> str:
        """Return display label for a vertex: name if available, else v{idx}."""
        if self.topic_names is not None and v in self.topic_names:
            return str(self.topic_names[v])
        return f"v{v}"

    def __repr__(self) -> str:
        if not self.terms:
            return "TopicFormula(⊥)"
        if self.is_atomic():
            return f"TopicFormula({self._vertex_label(next(iter(self.terms[0])))})"
        term_strs = []
        for term in self.terms:
            if len(term) == 1:
                term_strs.append(self._vertex_label(next(iter(term))))
            else:
                vertices = sorted(term)
                term_strs.append("(" + " ∧ ".join(self._vertex_label(v) for v in vertices) + ")")
        return "TopicFormula(" + " ∨ ".join(term_strs) + ")"


def simplify_formula(
    formula: TopicFormula,
    adjacency_closure: np.ndarray,
) -> TopicFormula:
    """
    Simplify a formula by removing redundant vertices and terms.
    
    For a tree, if v is an ancestor of w (v can reach w), then v ∧ w = w.
    So we keep the descendant and remove the ancestor.
    
    For disjunction, if term A ⊂ term B (proper subset), then A ∨ B = A.
    So we remove supersets.
    
    Parameters
    ----------
    formula : TopicFormula
        The formula to simplify.
    adjacency_closure : np.ndarray
        Transitive closure matrix where adjacency_closure[i, j] = 1
        means i is an ancestor of j (i can reach j).
    
    Returns
    -------
    TopicFormula
        Simplified formula.
    """
    if not formula.terms:
        return formula
    
    # Step 1: Simplify each term by removing ancestors
    simplified_terms = []
    for term in formula.terms:
        if not term:
            continue
        
        # Keep vertex v only if no other vertex in term is its descendant
        clean_term = set()
        for v in term:
            # Check if any other vertex is a descendant of v
            is_ancestor_of_something = False
            for u in term:
                if u != v and adjacency_closure[v, u]:
                    # v is ancestor of u, so v ∧ u = u, remove v
                    is_ancestor_of_something = True
                    break
            if not is_ancestor_of_something:
                clean_term.add(v)
        
        if clean_term:
            simplified_terms.append(clean_term)
    
    # Step 2: Remove terms that are supersets of other terms
    # If term_i ⊃ term_j, then term_i ∨ term_j = term_j, remove term_i
    final_terms = []
    for i, term_i in enumerate(simplified_terms):
        is_superset = False
        for j, term_j in enumerate(simplified_terms):
            if i != j and term_j < term_i:  # term_j is proper subset of term_i
                is_superset = True
                break
        if not is_superset:
            final_terms.append(term_i)
    
    # Step 3: Remove duplicates (same set of vertices)
    unique_terms = []
    seen = set()
    for term in final_terms:
        term_tuple = tuple(sorted(term))
        if term_tuple not in seen:
            seen.add(term_tuple)
            unique_terms.append(term)
    
    return TopicFormula(unique_terms)
