# задания1-3 обнова
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from scipy.stats import norm, multivariate_normal
import numpy as np

# Задание 1: Метод максимального правдоподобия


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    def neg_log_likelihood(params):
        tau, mu1, sigma1, mu2, sigma2 = params
        sigma1 = np.abs(sigma1)
        sigma2 = np.abs(sigma2)
        p1 = tau * norm.pdf(x, mu1, sigma1)
        p2 = (1 - tau) * norm.pdf(x, mu2, sigma2)
        return -np.sum(np.log(p1 + p2))

    initial_params = [tau, mu1, sigma1, mu2, sigma2]
    bounds = [(0, 1), (None, None), (0, None), (None, None), (0, None)]
    result = minimize(neg_log_likelihood, initial_params,
                      method='L-BFGS-B', bounds=bounds, tol=rtol)

    if not result.success:
        raise ValueError('Оптимизация не сходится: ' + result.message)
    return result.x

# Задание 2: EM-алгоритм для двойного гауссовского распределения


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3, max_iter=100):
    n = x.shape[0]
    gamma = np.zeros((n, 2))
    params = np.array([tau, mu1, sigma1, mu2, sigma2])
    for _ in range(max_iter):
        # E-шаг
        w1 = tau * norm.pdf(x, mu1, sigma1)
        w2 = (1 - tau) * norm.pdf(x, mu2, sigma2)
        gamma[:, 0] = w1 / (w1 + w2)
        gamma[:, 1] = w2 / (w1 + w2)

        # M-шаг
        tau = np.mean(gamma[:, 0])
        mu1 = np.sum(gamma[:, 0] * x) / np.sum(gamma[:, 0])
        mu2 = np.sum(gamma[:, 1] * x) / np.sum(gamma[:, 1])
        sigma1 = np.sqrt(
            np.sum(gamma[:, 0] * (x - mu1)**2) / np.sum(gamma[:, 0]))
        sigma2 = np.sqrt(
            np.sum(gamma[:, 1] * (x - mu2)**2) / np.sum(gamma[:, 1]))

        new_params = np.array([tau, mu1, sigma1, mu2, sigma2])
        if np.allclose(params, new_params, rtol=rtol):
            break
        params = new_params

    return params


# Задание 3: EM-алгоритм для двух двумерных гауссовских распределений


def em_double_2d_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3, max_iter=100):
    n = x.shape[0]
    gamma = np.zeros((n, 2))
    prev_log_likelihood = None

    for iter_count in range(max_iter):
        pdf1 = multivariate_normal.pdf(x, mean=mu1, cov=sigma1)
        pdf2 = multivariate_normal.pdf(x, mean=mu2, cov=sigma2)
        gamma[:, 0] = tau * pdf1
        gamma[:, 1] = (1 - tau) * pdf2
        gamma /= gamma.sum(1, keepdims=True)

        tau = gamma[:, 0].mean()
        mu1 = np.average(x, weights=gamma[:, 0], axis=0)
        mu2 = np.average(x, weights=gamma[:, 1], axis=0)
        sigma1 = np.cov(x.T, aweights=gamma[:, 0])
        sigma2 = np.cov(x.T, aweights=gamma[:, 1])

        log_likelihood = np.sum(
            np.log(pdf1 * gamma[:, 0] + pdf2 * gamma[:, 1]))

        if prev_log_likelihood is not None and abs(log_likelihood - prev_log_likelihood) < rtol:
            break
        prev_log_likelihood = log_likelihood

    return tau, mu1, sigma1, mu2, sigma2
