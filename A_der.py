import numpy as np
from utils import out_to_file
from task3 import X, Y, p, n, Ny, u_l, u_r, lamda, f
from task3 import u_l, u_r, psi

import numpy as np
import matplotlib.pyplot as plt

sigma_0 = 5.67e-8

def h(u):
    return sigma_0 * np.abs(np.power(u, 3))*u

def h_der(u):
    return 4 * sigma_0 * np.power(u, 3)

class A:
    def __init__(self):
        self.Nx = n * p
        self.Ny = Ny

        self.hx = X / (n * (p - 1))  # Шаг сетки по x
        self.hy = Y / (Ny - 1)  # Шаг сетки по y

    def get_grid_func(self, func, i, j):
        d = i // p

        return func((i) * self.hx, j * self.hy)

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
    # def Au(self, grid, i, j):
    #     if i == 0:
    #         return
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
        # print(res)
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


class A_der:

    def __init__(self):
        self.Nx = n * p
        self.Ny = Ny

        self.hx = X / (n * (p - 1))  # Шаг сетки по x
        self.hy = Y / (Ny - 1)  # Шаг сетки по y

    def get_grid_func(self, func, i, j):
        d = i // p
        return func(i * self.hx, j * self.hy)

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

        # print(res)

# print(grid)
# S = A_der(hx, hy)
# S.apply_operator()
# S = A(hx, hy)
# S.apply_operator()

