# задание 4 проба
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from astropy.coordinates import SkyCoord
from astroquery.utils.tap import TapPlus

# Создание экземпляра TapPlus для доступа к TAP-сервису
gaia = TapPlus(url="https://gaia.aip.de/tap", verbose=True)

# Настройки для загрузки данных
limit_magnitude = 16
center_coord = SkyCoord('02h21m00s +57d07m42s')
target_size = 0.5
range_ra = (center_coord.ra.degree - target_size / np.cos(center_coord.dec.radian),
            center_coord.ra.degree + target_size / np.cos(center_coord.dec.radian))
range_dec = (center_coord.dec.degree - target_size,
             center_coord.dec.degree + target_size)

# Формирование SQL-запроса
query = f'''
SELECT ra, dec, pmra, pmdec FROM gaiadr2.gaia_source
WHERE phot_bp_mean_mag < {limit_magnitude} AND pmra IS NOT NULL AND pmdec IS NOT NULL
AND ra BETWEEN {range_ra[0]} AND {range_ra[1]} AND dec BETWEEN {range_dec[0]} AND {range_dec[1]}
'''

# Выполнение асинхронного запроса к серверу Gaia
job = gaia.launch_job_async(query)
stars = job.get_results()
pmra = np.asarray(stars['pmra'])
pmdec = np.asarray(stars['pmdec'])

# Данные для анализа
data = np.column_stack((pmra, pmdec))

# Инициализация параметров EM-алгоритма
n_clusters = 2
weights = np.full(n_clusters, 1 / n_clusters)
means = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
covariances = np.array([np.cov(data, rowvar=False)] * n_clusters)

# EM-алгоритм
max_iter = 100
for _ in range(max_iter):
    # E-Step
    responsibilities = np.array([weights[i] * multivariate_normal.pdf(data, mean=means[i], cov=covariances[i])
                                 for i in range(n_clusters)]).T
    responsibilities /= responsibilities.sum(1)[:, np.newaxis]

    # M-Step
    for i in range(n_clusters):
        weight = responsibilities[:, i].sum()
        means[i] = np.dot(responsibilities[:, i], data) / weight
        diff = data - means[i]
        covariances[i] = np.dot(responsibilities[:, i] * diff.T, diff) / weight
        weights[i] = weight / data.shape[0]

# Визуализация результатов
plt.figure(figsize=(10, 5))
plt.scatter(data[:, 0], data[:, 1], c=responsibilities[:, 0],
            cmap='viridis', alpha=0.6)
plt.title('Результаты кластеризации EM-алгоритма')
plt.xlabel('PMRA')
plt.ylabel('PMDEC')
plt.colorbar(label='Принадлежность к первому кластеру')
plt.show()
