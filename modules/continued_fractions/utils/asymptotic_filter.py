def is_asymptotically_convergent(a_deg, a_leading_coef, b_deg, b_leading_coef, strict=False):
    """
    Applies algebraic constraint checks (e.g., Worpitzky's theorem and Poincaré's theorem)
    to predict if a Generalized Continued Fraction will mathematically converge,
    avoiding the computational cost of evaluating the series.

    :param a_deg: Degree of the P(n) sequence.
    :param a_leading_coef: Leading coefficient of P(n).
    :param b_deg: Degree of the Q(n) sequence.
    :param b_leading_coef: Leading coefficient of Q(n).
    :param strict: If true, discards boundaries where convergence depends on sub-dominant terms.
    :return: True if the mathematical constraints permit general convergence.
    """
    # 1. Degree Balance Condition
    # If B grows more than quadratically faster than A, the fraction diverges.
    if a_deg * 2 < b_deg:
        return False
        
    # 2. Sign Consistency
    # To ensure a stable non-alternating limit, a_n should be positive for large n.
    # (Negative leading coefficients can be normalized, but we filter them out to avoid duplicates)
    if a_leading_coef <= 0:
        return False
        
    # 3. Balanced Degree Rigorous Checks (Poincaré's Theorem)
    if a_deg * 2 == b_deg:
        # For q_n = a * q_{n-1} + b * q_{n-2}, the characteristic polynomial is t^2 - at - b = 0.
        # Worpitzky boundary: discriminant >= 0 ensures roots are real.
        # If discriminant < 0, roots are complex conjugates, so |r1| == |r2| -> Divergence.
        discriminant = (a_leading_coef ** 2) + 4 * b_leading_coef
        
        if discriminant < 0:
            return False
            
        if strict and discriminant == 0:
            # Roots are real but equal magnitude (r1 == r2) -> Convergence depends on lower terms.
            return False
            
        # Poincaré ratio condition implies we need |r1| != |r2|.
        # We already handled r1 == r2 (discriminant == 0).
        # We must also handle r1 == -r2, which implies a_leading_coef == 0.
        # But we already enforce a_leading_coef > 0 above, so this is safely covered.

    return True
