import numpy as np

from src.greeks import bs_delta, bs_vega, finite_difference_delta, finite_difference_vega


def test_delta_matches_finite_difference():
    S = np.array([100.0, 120.0, 95.0])
    K = np.array([100.0, 110.0, 90.0])
    T = np.array([0.5, 1.0, 0.25])
    sigma = np.array([0.2, 0.25, 0.3])
    is_call = np.array([True, False, True])
    r, q = 0.05, 0.0

    analytic = bs_delta(S, K, T, r, q, sigma, is_call)
    numeric = finite_difference_delta(S, K, T, r, q, sigma, is_call)
    assert np.allclose(analytic, numeric, atol=1e-4)


def test_vega_matches_finite_difference():
    S = np.array([100.0, 120.0, 95.0])
    K = np.array([100.0, 110.0, 90.0])
    T = np.array([0.5, 1.0, 0.25])
    sigma = np.array([0.2, 0.25, 0.3])
    is_call = np.array([True, False, True])
    r, q = 0.05, 0.0

    analytic = bs_vega(S, K, T, r, q, sigma)
    numeric = finite_difference_vega(S, K, T, r, q, sigma, is_call)
    assert np.allclose(analytic, numeric, atol=1e-3)

