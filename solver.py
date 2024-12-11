from A_der import A, A_der
from task1 import N_x
import numpy as np
import time
from task1 import Lx, Ly, p, n, u_l, u_r, lamda, sigma_0
import matplotlib.pyplot as plt

class Solver:
    def __init__(self, Ny):
        self.Nx = N_x
        self.Ny = Ny
        self.A = A(Ny)
        self.A_d = A_der(Ny)

        self.p = np.ones((self.Nx, self.Ny))

        self.hx = Lx / (n * (p - 1))  # Шаг сетки по x
        self.hy = Ly / (Ny - 1)  # Шаг сетки по y

    def init(self, u):
        for i in range(self.Nx):
            u[i][0] = 200.
            u[i][self.Ny-1] = 200.

            # Усредняем значения по столбцам между краями
        for j in range(1, self.Ny - 1):  # Проходим по всем столбцам между краями
            for i in range(self.Nx):  # Проходим по всем строкам
                u[i][j] = (u[i][0] + u[i][self.Ny - 1]) / 2.  # Усреднение

    def sc_mult(self, u, v):
        s = 0
        # k = 0
        # for i in range(1, self.Nx - 1):
        #     k = 0
        #     for j in range(1, self.Ny - 1):
        #         k += u[i, j] * v[i, j] * self.hy
        #     s+= k * self.hx
        for i in range(0, self.Nx):
            for j in range(1, self.Ny - 1):
                s += u[i, j] * v[i, j]
        return s

    def BiCGStab(self, b, tol=1e-3, max_time=300, max_iter=10000000):
        start_time = time.time()

        p_solv = np.ones((self.Nx, self.Ny))
        for i in range(self.Nx):
            p_solv[i][0] = 0
            p_solv[i][self.Ny - 1] = 0

        #self.init(p_solv)
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

        delta = self.BiCGStab(-self.A.apply_operator(self.p))
        self.p += delta
        while(self.sc_mult(delta, delta)**(1/2) > 1e-1):
            print(self.sc_mult(delta, delta)**(1/2))
            delta = self.BiCGStab(-self.A.apply_operator(self.p))
            self.p+=delta
        return self.p

    def plot_heatmap(self):
        """
        Построение тепловой карты для решения
        """
        plt.figure(figsize=(8, 6))
        plt.imshow(self.p.T, extent=[0, 1. , 0, 1.], origin = 'lower', cmap='inferno', aspect='auto')
        plt.colorbar(label='Температура')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.title(f'Норма невязки: {self.get_r_norm()}')
        # output_path = os.path.join('out_im_bi', 'heatmap_' + str(self.Nx) +'_'+ str(self.M.TaskNumber) +'.png')
        # plt.savefig(output_path)
        plt.show()
