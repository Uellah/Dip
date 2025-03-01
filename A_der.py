import numpy as np
from utils import out_to_file
from task6 import X, Y, p, n, Ny, lamda
from task6 import u_up, u_down, u_l, u_r, psi, f, u_an

import numpy as np
import matplotlib.pyplot as plt

sigma_0 = 5.67e-8

def h(u):
    return sigma_0 * np.abs(np.power(u, 3))*u

def h_der(u):
    return 4 * sigma_0 * np.power(u, 3)

class Grid():
    def __init__(self):
        self.X = X
        self.Y = Y

        self.Nx = n * p
        self.Ny = Ny

        self.hx = X / (n * (p - 1))  # Шаг сетки по x
        self.hy = Y / (Ny - 1)  # Шаг сетки по y

    def get_grid_func(self, func, i, j):
        d = i // p
        return func(i * self.hx, j * self.hy)

    def apply_border(self, u):
        for i in range(self.Nx):
            u[i][0] = self.get_grid_func(u_down, i,0)
            u[i][-1] = self.get_grid_func(u_up, i, self.Ny - 1)

    def init(self, u):
        self.apply_border(u)

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

    def get_analytical_solve(self):
        tmp = np.zeros((self.Nx, self.Ny))
        for i in range(self.Nx):
            for j in range(self.Ny):
                tmp[i][j] = self.get_grid_func(u_an, i, j)
        return tmp

    def get_r_norm(self, solve, an_solve):
        return np.linalg.norm(solve - an_solve)


class A(Grid):
    def __init__(self):
        super().__init__()

    def second_y_der(self, u, i, j):
        res = (u[i][j-1] - 2 * u[i][j] + u[i][j+1])/np.power(self.hy, 2)
        return res

    def second_x_der(self, u, i, j):
        res = (u[i-1][j] - 2 * u[i][j] + u[i+1][j])/np.power(self.hx, 2)
        return res

    def puasson(self, u, i, j):
        return self.second_x_der(u, i, j) + self.second_y_der(u, i, j)

    def left_bound_exter(self, u, j):
        first = (u[1][j] - u[0][j])/np.power(self.hx, 2)
        second = self.second_y_der(u, 0, j) / 2.
        # print(self.get_grid_func(u_l, 0, j))
        third = (h(self.get_grid_func(u_l, 0, j)) - h(u[0][j]))/self.hx/lamda
        return -first - second - third - self.get_grid_func(f, 0, j) / 2. / lamda

    def right_bound_exter(self, u, i, j):
        first = (u[i - 1][j] - u[i][j])/np.power(self.hx, 2)
        second = self.second_y_der(u, i, j) / 2.
        # print("  hhh ", self.get_grid_func(u_r, 0, j))
        third = (h(self.get_grid_func(u_r, i, j)) - h(u[i][j])) / self.hx / lamda
        return - first - second - third - self.get_grid_func(f, i, j) / 2. / lamda

    def left_bound_inner(self, u, i, j):
        first = (u[i - 1][j] - u[i][j]) / np.power(self.hx, 2)
        second = self.second_y_der(u, i, j) / 2.
        third = (h(u[i+1][j])- h(u[i][j]) + self.get_grid_func(psi, i, j)) / self.hx / lamda
        return - first - second - third - self.get_grid_func(f, i, j) / 2. / lamda

    def right_bound_inner(self, u, i, j):
        first = (u[i+1][j] - u[i][j]) / np.power(self.hx, 2)
        second = self.second_y_der(u, i, j) / 2.
        third = (h(u[i-1][j]) - h(u[i][j]) - self.get_grid_func(psi, i, j)) / self.hx / lamda
        return - first - second - third - self.get_grid_func(f, i, j) / 2. / lamda

    def apply_operator(self, grid):
        res = np.zeros((self.Nx, self.Ny))
        # res = grid.copy()
        for j in range(1, self.Ny - 1):
            res[0][j] = self.left_bound_exter(grid, j)
            res[-1][j] = self.right_bound_exter(grid, self.Nx - 1, j)

        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                if i % p == 0:
                    res[i][j] = self.left_bound_inner(grid, i, j)
                    continue
                if (i % p) == (p - 1):
                    res[i][j] = self.right_bound_inner(grid, i,  j)
                    continue
                res[i][j] = - self.puasson(grid, i, j) - self.get_grid_func(f, i, j) / lamda
        return res

    def plot_operator_application(self):
        grid_visual = np.zeros((self.Nx, self.Ny))

        for j in range(1, self.Ny - 1):
            grid_visual[0][j] = 1  # Левый внешний край
            grid_visual[-1][j] = 2  # Правый внешний край

        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                if i % p == 0:
                    grid_visual[i][j] = 3  # Внутренний левый край
                    continue
                if (i % p) == (p - 1):
                    grid_visual[i][j] = 4  # Внутренний правый край
                    continue
                grid_visual[i][j] = 5  # Остальные точки

        plt.figure(figsize=(8, 8))
        plt.imshow(grid_visual.T, cmap="coolwarm", origin="lower")

        cbar = plt.colorbar()
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.set_ticklabels([
            "Левый внешний край",
            "Правый внешний край",
            "Внутренний левый край",
            "Внутренний правый край",
            "Внутренние точки"
        ])

        for i in range(self.Ny):
            for j in range(self.Nx):
                plt.gca().add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor='black', linewidth=0.5))
        plt.title("Применение операторов в сетке")
        plt.xlabel("j (ось Y)")
        plt.ylabel("i (ось X)")
        plt.show()


class A_der(Grid):

    def __init__(self):
        super().__init__()

    def second_y_der(self, u, i, j):
        res = (u[i][j - 1] - 2 * u[i][j] + u[i][j + 1]) / np.power(self.hy, 2)
        return res

    def second_x_der(self, u, i, j):
        res = (u[i - 1][j] - 2 * u[i][j] + u[i + 1][j]) / np.power(self.hx, 2)
        return res

    def puasson(self, u, i, j):
        return self.second_x_der(u, i, j) + self.second_y_der(u, i, j)

    def left_bound_exter(self, u, u_der, j):
        first = (1/np.power(self.hy, 2) + 1/np.power(self.hx, 2)  + h_der(u[0][j])/self.hx/lamda)*u_der[0][j]
        second = 1/(2. * np.power(self.hy, 2))*(u_der[0][j-1] + u_der[0][j+1])
        third = u_der[1][j]/np.power(self.hx, 2)
        return first - second - third

    def right_bound_exter(self, u, u_der, i, j):
        first = (1 / np.power(self.hy, 2) + 1 / np.power(self.hx, 2) + h_der(u[i][j]) / self.hx / lamda) * u_der[i][j]
        second = 1 / (2 * np.power(self.hy, 2)) * (u_der[i][j - 1] + u_der[i][j + 1])
        third = u_der[i - 1][j] / np.power(self.hx, 2)
        return first - second - third

    def left_bound_inner(self, u, u_der, i, j):
        first = (1 / np.power(self.hy, 2) + 1 / np.power(self.hx, 2) + h_der(u[i][j]) / self.hx / lamda) * u_der[i][j]
        second = 1 / (2 * np.power(self.hy, 2)) * (u_der[i][j - 1] + u_der[i][j + 1])
        third = h_der(u[i + 1][j]) / self.hx / lamda * u_der[i + 1][j]
        return first - second - third - u_der[i - 1][j] / np.power(self.hx, 2)

    def right_bound_inner(self, u, u_der, i, j):
        first = (1 / np.power(self.hy, 2) + 1 / np.power(self.hx, 2) + h_der(u[i][j]) / self.hx / lamda) * u_der[i][j]
        second = 1 / (2 * np.power(self.hy, 2)) * (u_der[i][j - 1] + u_der[i][j + 1])
        third = h_der(u[i - 1][j]) / self.hx / lamda * u_der[i - 1][j]
        return first - second - third - u_der[i + 1][j] / np.power(self.hx, 2)

    def apply_operator(self, u, u_der):
        res = np.zeros((self.Nx, self.Ny))
        #res = u_der.copy()
        for j in range(1, self.Ny - 1):
            res[0][j] = self.left_bound_exter(u, u_der, j)
            res[-1][j] = self.right_bound_exter(u, u_der, self.Nx - 1, j)

        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                if i % p == 0:
                    res[i][j] = self.left_bound_inner(u, u_der, i, j)
                    continue
                if (i % p) == (p - 1):
                    res[i][j] = self.right_bound_inner(u, u_der, i, j)
                    continue
                res[i][j] = - self.puasson(u_der, i, j)
        return res