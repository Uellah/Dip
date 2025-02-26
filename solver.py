from A_der import A, A_der

import numpy as np
import time
import matplotlib.pyplot as plt

from task3 import X, Y, p, n, Ny
from task3 import u_up, u_down

class Solver:
    def __init__(self):
        self.Nx = n * p
        self.Ny = Ny

        self.A = A()
        self.A_d = A_der()

        self.p = np.full((self.Nx, self.Ny), 500.)

        self.hx = X / (n * (p - 1))  # Шаг сетки по x
        self.hy = Y / (Ny - 1)  # Шаг сетки по y

    def get_grid_func(self, func, i, j):
        d = i // p
        return func((i - d) * self.hx, j * self.hy)

    def apply_border(self, u):
        for i in range(self.Nx):
            u[i][0] = self.get_grid_func(u_down, i,0)
            u[i][-1] = self.get_grid_func(u_up, i, self.Ny - 1)

    def init(self, u):
        for i in range(self.Nx):
            u[i][0] = self.get_grid_func(u_down, i,0)
            u[i][-1] = self.get_grid_func(u_up, i, self.Ny - 1)

            # Усредняем значения по столбцам между краями
        for i in range(self.Nx):  # Проходим по всем столбцам между краями
            tmp = u[i][self.Ny - 1] - u[i][0]
            for j in range(1, self.Ny - 1):  # Проходим по всем строкам
                u[i][j] = u[i][0] + j / (self.Ny - 2) * tmp # Усреднение

    def sc_mult(self, u, v):
        s = 0
        for i in range(0, self.Nx):
            for j in range(1, self.Ny - 1):
                if i % p == 0 or i % p == p - 1:
                    s += 0.5 * u[i, j] * v[i, j]
                else:
                    s += u[i, j] * v[i, j]
        return s

    def BiCGStab(self, start, b, tol=1e-3, max_time=100, max_iter=10000000):
        start_time = time.time()

        p_solv = start

        r0 = np.zeros((self.Nx, self.Ny))

        nach = self.A_d.apply_operator(self.p, p_solv)

        for i in range(0, self.Nx):
            for j in range(1, self.Ny - 1):
                r0[i, j] = b[i][j] - nach[i][j]

        r = r0.copy()
        rho_old = alpha = omega = 1.0
        v = np.zeros_like(r0)
        p = np.zeros_like(r0)

        iter_count = 0

        while iter_count < max_iter and (time.time() - start_time) < max_time:
            rho_new = self.sc_mult(r0, r)
            if abs(rho_new) < 1e-14:
                print("Прерывание: rho слишком мал.")
                break
            if iter_count == 0:
                p = r.copy()
            else:
                beta = (rho_new / rho_old) * (alpha / omega)
                p = r + beta * (p - omega * v)

            v = self.A_d.apply_operator(self.p, p)
            alpha = rho_new / self.sc_mult(r0, v)

            s = r - alpha * v

            if np.linalg.norm(s) < tol:
                p_solv += alpha * p
                #print(f"Сошелся за {iter_count} итераций")
                break

            t = self.A_d.apply_operator(self.p, s)
            omega = self.sc_mult(t, s) / self.sc_mult(t, t)

            p_solv += alpha * p + omega * s
            r = s - omega * t

            if np.linalg.norm(r) < tol:
                #print(f"Сошелся за {iter_count} итераций")
                break

            rho_old = rho_new
            iter_count += 1

        return p_solv

    def solve(self):
        self.init(self.p)

        tmp = np.ones((self.Nx, self.Ny), dtype = 'double')
        for i in range(self.Nx):
            tmp[i][0] = 0.
            tmp[i][-1] = 0.
        iter = 0

        delta = self.BiCGStab(tmp,-self.A.apply_operator(self.p))
        self.p += delta
        while self.sc_mult(delta, delta) > 0.01:

            print(self.sc_mult(delta, delta)**(1/2))
            # if iter % 10 == 0:
            #     delta = tmp
            old_d = delta
            delta = self.BiCGStab(old_d, -self.A.apply_operator(self.p))
            self.p += delta
            iter += 1
        return self.p

    def plot_heatmap(self, other_array):
        """
        Построение тепловых карт для решения.
        Слева отображается self.p с подписью "Мой метод",
        справа – переданный массив other_array с подписью "Встроенный метод".

        Аргументы:
            other_array (numpy.ndarray): Массив такого же размера, как self.p, для сравнения.
        """
        import matplotlib.pyplot as plt

        # Создаем фигуру с двумя подграфиками (subplot'ами)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Первый subplot: наш метод
        im0 = axs[0].imshow(self.p.T, extent=[0, X, 0, Y], origin='lower', cmap='jet', aspect='auto')
        axs[0].set_title('Мой метод')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        fig.colorbar(im0, ax=axs[0], label='Temp')

        # Второй subplot: встроенный метод
        im1 = axs[1].imshow(other_array.T, extent=[0, X, 0, Y], origin='lower', cmap='jet', aspect='auto')
        axs[1].set_title('Встроенный метод')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        fig.colorbar(im1, ax=axs[1], label='Temp')

        plt.show()


