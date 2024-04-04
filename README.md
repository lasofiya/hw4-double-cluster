# Домашнее задание № 4
## Восстановление параметров статистического распределения

В этом задании вы будете восстанавливать параметры смешанных статистических распределений, используя два метода: метод максимального правдоподобия и EM-метод. Научным заданием будет выделение двух рассеянных скоплений  [h и χ Персея](https://apod.nasa.gov/apod/ap091204.html) в звёздном поле.

**Дедлайн 18 апреля в 23:55**

Вы должны реализовать следующие алгоритмы в файле `mixfit.py`:

1. **Метод максимального правдоподобия для смеси двух одномерных нормальных распределений.** Напишите функцию `max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3)`, где `x` — массив данных, остальные позиционные аргументы — начальные приближения для искомых параметров распределения, `rtol` — относительная точность поиска параметров, функция должна возвращать кортеж из параметров распределения в том же порядке, что и в сигнатуре функции. Для оптимизации разрешается использовать `scipy.optimize`.

2. **[Expectation-maximization метод](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm)** для той же задачи: `em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3)`.

3. **EM-метод для смеси двух двумерных нормальных распределений** — $\tau N(\mu_1, \Sigma_1) + (1-\tau) N(\mu_2, \Sigma_2)$. Напишите функцию `em_double_2d_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3)`.
  * $\tau$ (`tau`) — относительное количество звезд в скоплениях.
  * $\mu_1$ (`mu1`) — двумерный вектор, средняя скорость звезд скоплений.
  * Диагональная матрица 2x2, элементы которой равны $\sigma_1^2$ (`sigma1`). Разброс собственных движений звезд скоплений.
  ```math
  \Sigma_1 = \begin{bmatrix}
  \sigma_1^2 & 0 \\
  0 & \sigma_1^2
  \end{bmatrix}
  ```
  * $\mu_2$ (`mu2`) — двумерный вектор, средняя скорость звезд поля.
  * Диагональная матрица 2x2, элементы которой равны $\sigma_2^2$ (`sigma2`). Разброс собственных движений звезд поля.
  ```math
  \Sigma_2 = \begin{bmatrix}
  \sigma_2^2 & 0 \\
  0 & \sigma_2^2
  \end{bmatrix}
  ```

4. Примените EM-метод для смеси двух двумерных нормальных распределений в файле `per.py` для решения задачи нахождения скоплений h и χ Персея.
Вам понадобиться модуль [astroquery.gaia](https://astroquery.readthedocs.io/en/latest/gaia/gaia.html) для того, чтобы загрузить координаты звёзд поля.
Координаты центра двойного скопления `02h21m00s +57d07m42s`, используйте поле размером `1.0 x 1.0` градус.
Пример запроса для получения данных:

   ```python
   limit_magnitude = 16
   center_coord = SkyCoord('02h21m00s +57d07m42s')
   target_size = 0.5
   range_ra = (center_coord.ra.degree - target_size / np.cos(center_coord.dec.radian),
               center_coord.ra.degree + target_size / np.cos(center_coord.dec.radian))
   range_dec = (center_coord.dec.degree - target_size, center_coord.dec.degree + target_size)
   query = '''
   SELECT ra, dec, pmra, pmdec FROM gaiadr3.gaia_source WHERE phot_bp_mean_mag < {:.2f} AND pmra IS NOT NULL AND pmdec IS NOT NULL AND ra BETWEEN {:} AND {:} AND dec BETWEEN {:} AND {:}
   '''.format(limit_magnitude, *range_ra, *range_dec)

   job = Gaia.launch_job_async(query)
   stars = job.get_results()
   ra = np.asarray(stars['ra']._data)
   dec = np.asarray(stars['dec']._data)
   pmra = np.asarray(stars['pmra']._data)
   pmdec = np.asarray(stars['pmdec']._data)
   ```

5. В файл `per.json` сохраните результаты в следующем формате:

    ```json
    {
      "ratio": 0.4,
      "motion": {
        "cluster": {"ra": -0.63, "dec": -1.15},
        "background": {"ra": 1.53, "dec": 2.05}
      }
    }
    ```

6. В файле `per.png` изобразите график рассеяния точек звёздного поля, и график рассеяния собственных движений.
На обоих графиках для каждой точки отметьте цветом условную вероятность принадлежности к скоплениям (используйте вычисленные параметры `T`).
На графике рассеяния собственных движения (скоростей) обозначьте скопления окружностью с центром в центре распределения и радиусом равным стандартном отклонению (корень из дисперсии).

> **Задания на бонусные баллы.**

7. **EM-метод для смеси трех нормальных распределений** — $\tau_1 N(\mu_1, \Sigma) + \tau_2 N(\mu_2, \Sigma) + (1-\tau_1-\tau_2) N(0, \Sigma_0)$
Напишите функцию `em_double_cluster(x, tau1, tau2, muv, mu1, mu2, sigma02, sigmax2, sigmav2, rtol=1e-5)`, где
  * $x$ (x) — массив `(N, 4)` состоящий из двух столбцов координат звезд, и двух столбцов собственых движений звезд.
[Собственными движениями](http://www.astronet.ru/db/msg/1171379) называется скорость движения звезды в [картинной плоскости](http://www.astronet.ru/db/msg/1190817/node7.html) в угловых единицах;
  * $\tau_1$ (`tau1`), $\tau_2$ (`tau2`) — относительные количества звезд в первом и втором скоплениях. Обратите внимание, что относительное количество звезд поля может быть найдена из условия нормировки $1-\tau_1-\tau_2$.
  * `mu1` — четырехмерный параметр распределения звезд в первом скоплении. Состоит из вектора среднего положения звезд скопления $\mu_{x,1}$ и вектора среднего собственного движения скопления $\mu_v$.
  ```math
  \mu_1 = \begin{pmatrix}
  \mu_{x,1} \\
  \mu_{v}
  \end{pmatrix}
  ```
  * `mu2` — четырехмерный параметр распределения звезд во втором скоплении. Состоит из вектора среднего положения звезд скопления $\mu_{x,2}$ и вкутора среднего собственного движения скопления $\mu_v$. Обратите внимание, что предполагается, что скорость движения скоплений — одинаковая.
  ```math
  \mu_2 = \begin{pmatrix}
  \mu_{x,2} \\
  \mu_{v}
  \end{pmatrix}
  ```
  * Диагональная матрица 4x4. Первые два компоненты которой равны $\sigma^2_x$ (`sigmax2`) и отвечают за разброс положения звезд скоплений вокруг среднего,  а последние два компонента $\sigma^2_v$ (`sigmav2`) отвечают за разброс собственных движений звезд скоплений относительно среднего. Иначе говоря, предполагается, что разброс по координатам имеет симметричный характер и одинаковую дисперсию по обоим направлениям. Аналогичное предположение действует для разброса проекций скоростей.
  ```math
  \Sigma = \begin{bmatrix}
  \sigma^2_x & 0 & 0 & 0 \\
  0 & \sigma^2_x & 0 & 0 \\
  0 & 0 & \sigma^2_v & 0 \\
  0 & 0 & 0 & \sigma^2_v
  \end{bmatrix}
  ```
  * Диагональная матрица 2x2, элементы которой равны $\sigma_0^2$ (`sigma02`). Разброс собственных движений звезд поля. Обратите внимание, что предполагается, что среднее собственное движение звезд поля — нулевое.
  ```math
  \Sigma_0 = \begin{bmatrix}
  \sigma_0^2 & 0 \\
  0 & \sigma_0^2
  \end{bmatrix}
  ```

8. Примените этот усовершенствованный EM-метод в файле `per.py` для решения задачи нахождения центров и относительного числа звёзд в скоплениях h и χ Персея.

9. В файл `double_per.json` сохрнаите результаты:

    ```json
    {
      "size_ratio": 1.2,
      "motion": {"ra": -0.63, "dec": -1.15},
      "clusters": [
        {
          "center": {"ra": 35.35, "dec": 57.07},
        },
        {
          "center": {"ra": 36.36, "dec": 57.57},
        }
      ]
    }
    ```

10. В файле `double_per.png` изобразите график рассеяния точек звёздного поля, и график рассеяния собственных движений.
На обоих графиках для каждой точки отметьте цветом условную вероятность принадлежности к одному из скоплений (используйте вычисленные параметры `T1` и `T2`).
Обозначьте скопления окружностями с центром в центре распределения и радиусом равным стандартном отклонению (корень из дисперсии).
