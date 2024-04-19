# Задание7
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigma2, sigmav2, rtol=1e-5, max_iter=100, verbose=True):
    N = x.shape[0]  # Количество звезд

    # Инициализация ковариационных матриц
    sigma = np.diag([sigma2, sigma2, sigmav2, sigmav2])
    sigma0 = np.diag([sigma02, sigma02])

    # Инициализация относительных вероятностей
    tau = np.array([tau1, tau2, 1 - tau1 - tau2])

    # Инициализация распределений
    cluster1 = multivariate_normal(mean=mu1, cov=sigma)
    cluster2 = multivariate_normal(mean=mu2, cov=sigma)
    field = multivariate_normal(mean=muv, cov=sigma0)

    # Для хранения истории параметров
    history = {'tau': [], 'mu1': [], 'mu2': [],
               'sigma': [], 'log_likelihood': []}

    for iteration in range(max_iter):
        # E-шаг
        p1 = cluster1.pdf(x) * tau[0]
        p2 = cluster2.pdf(x) * tau[1]
        p0 = field.pdf(x[:, :2]) * tau[2]

        # Вычисление весов gamma
        weights = np.vstack((p1, p2, p0)).T
        weights /= weights.sum(axis=1)[:, np.newaxis]

        # M-шаг
        # Обновление tau
        tau = weights.mean(axis=0)

        # Обновление mu1 и mu2
        mu1 = np.dot(weights[:, 0], x) / weights[:, 0].sum()
        mu2 = np.dot(weights[:, 1], x) / weights[:, 1].sum()

        # Обновление распределений
        cluster1 = multivariate_normal(mean=mu1, cov=sigma)
        cluster2 = multivariate_normal(mean=mu2, cov=sigma)

        # Обновление логарифмического правдоподобия
        log_likelihood = np.log(p1 + p2 + p0).sum()

        # Вывод промежуточных результатов
        if verbose:
            print(f"Iteration {iteration}, log likelihood: {log_likelihood}")

        # Сохранение истории
        history['tau'].append(tau.copy())
        history['mu1'].append(mu1.copy())
        history['mu2'].append(mu2.copy())
        history['sigma'].append(sigma.copy())
        history['log_likelihood'].append(log_likelihood)

        # Проверка сходимости
        if iteration > 0 and np.abs(history['log_likelihood'][-1] - history['log_likelihood'][-2]) < rtol:
            break

    return {'tau': tau, 'mu1': mu1, 'mu2': mu2, 'sigma': sigma, 'log_likelihood': log_likelihood, 'history': history}


# Создадим синтетические данные для тестирования
np.random.seed(0)
x_test = np.random.rand(100, 4)

# Пример использования функции
results = em_double_cluster(
    x_test, 0.3, 0.3, [0, 0], [1, 1, 0, 0], [2, 2, 1, 1], 0.1, 1, 0.5, verbose=True
)

# Распаковка результатов
tau1, tau2, mu1, mu2, sigma, log_likelihood, history = results['tau'][0], results['tau'][
    1], results['mu1'], results['mu2'], results['sigma'], results['log_likelihood'], results['history']

# Проверим результаты после завершения
print("Final estimated parameters:")
print("Tau1:", tau1)
print("Tau2:", tau2)
print("Mu1:", mu1)
print("Mu2:", mu2)
print("Sigma:", sigma)
print("Log likelihood:", log_likelihood)

# Визуализация сходимости логарифмического правдоподобия
log_likelihoods = history['log_likelihood']
plt.plot(log_likelihoods)
plt.title('Log Likelihood Convergence')
plt.xlabel('Iteration')
plt.ylabel('Log Likelihood')
plt.show()
