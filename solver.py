from A_der import A, A_der

import numpy as np
import time
import matplotlib.pyplot as plt

from task2 import X, Y, p, n, Ny
from task2 import u_up, u_down

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

    def BiCGStab(self, start, b, tol=1e-3, max_time=60, max_iter=10000000):
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

        delta = self.BiCGStab(tmp,-self.A.apply_operator(self.p))
        self.p += delta
        while self.sc_mult(delta, delta)**(1 / 2) > 1e-2:
            print(self.sc_mult(delta, delta)**(1/2))
            delta = self.BiCGStab(delta, -self.A.apply_operator(self.p))
            self.p+=delta
        return self.p

    def plot_heatmap(self):
        """
        Построение тепловой карты для решения
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.p.T, extent=[0, .25 , 0, .25], origin = 'lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Температура')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(f'Норма невязки: {self.get_r_norm()}')
        # output_path = os.path.join('out_im_bi', 'heatmap_' + str(self.Nx) +'_'+ str(self.M.TaskNumber) +'.png')
        # plt.savefig(output_path)
        plt.show()
