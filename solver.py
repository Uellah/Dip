from A_der import A, A_der, Grid

import numpy as np
import time
import matplotlib.pyplot as plt

class Solver(Grid):
    def __init__(self):
        super().__init__()
        self.A = A()
        self.A_d = A_der()
        self.p = np.full((self.Nx, self.Ny), 500.)

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

    def plot_heatmap(self, other_array = np.zeros((10, 10))):
        # Создаем фигуру с двумя подграфиками (subplot'ами)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Первый subplot: наш метод
        im0 = axs[0].imshow(self.p.T, extent=[0, self.X, 0, self.Y], origin='lower', cmap='jet', aspect='auto')
        axs[0].set_title('Решение')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('y')
        fig.colorbar(im0, ax=axs[0], label='Temp')

        # Второй subplot: встроенный метод
        im1 = axs[1].imshow(other_array.T, extent=[0, self.X, 0, self.Y], origin='lower', cmap='jet', aspect='auto')
        axs[1].set_title('Аналитическое решение')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        fig.colorbar(im1, ax=axs[1], label='Temp')

        plt.show()