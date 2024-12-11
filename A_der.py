import numpy as np
from utils import out_to_file
from task1 import Lx, Ly, p, n, u_l, u_r, lamda, sigma_0

def h(u):
    return sigma_0 * np.abs(np.power(u, 3))*u

def h_der(u):
    return 4 * sigma_0 * np.power(u, 3)

class A:
    def __init__(self, Ny):
        self.N_x = n * p
        self.Ny = Ny

        self.hx = Lx / (n * (p - 1))  # Шаг сетки по x
        self.hy = Ly / (Ny - 1)  # Шаг сетки по y

    def second_y_der(self, u, i, j):
        res = (u[i][j-1] - 2 * u[i][j] + u[i][j+1])/np.power(self.hy, 2)
        return res

    def second_x_der(self, u, i, j):
        res = (u[i-1][j] - 2 * u[i][j] + u[i+1][j])/np.power(self.hx, 2)
        return res

    def puasson(self, u, i, j):
        return self.second_x_der(u, i, j) + self.second_y_der(u, i, j)

    def left_bound_exter(self, u, j):
        first = (u[0][j] - u[1][j])/np.power(self.hx, 2)
        second = self.second_y_der(u, 0, j)/2
        third = (h(u_l) - h(u[0][j]))/self.hx/lamda
        return first - second - third

    def right_bound_exter(self, u, i, j):
        first = (u[i][j] - u[i-1][j])/np.power(self.hx, 2)
        second = self.second_y_der(u, i, j)/2
        third = (h(u_r) - h(u[i][j]))/self.hx/lamda
        return first - second - third

    def left_bound_inner(self, u, i, j):
        first = (u[i][j] - u[i - 1][j]) / np.power(self.hx, 2)
        second = self.second_y_der(u, i, j) / 2
        third = (h(u[i+1][j])- h(u[i][j])) / self.hx / lamda
        return first - second - third

    def right_bound_inner(self, u, i, j):
        first = (u[i][j] - u[i + 1][j]) / np.power(self.hx, 2)
        second = self.second_y_der(u, i, j) / 2
        third = (h(u[i-1][j]) - h(u[i][j])) / self.hx / lamda
        return first - second - third
    # def Au(self, grid, i, j):
    #     if i == 0:
    #         return
    def apply_operator(self, grid):
        res = np.zeros((self.N_x, self.Ny))
        # res = grid.copy()
        for j in range(1, self.Ny - 1):
            res[0][j] = self.left_bound_exter(grid, j)
            res[self.N_x - 1][j] = self.right_bound_exter(grid, self.N_x - 1, j)

        for i in range(1, self.N_x - 1):
            for j in range(1, self.Ny - 1):
                if i % p == 0:
                    res[i][j] = self.left_bound_inner(grid, i, j)
                    break
                if i % p == p - 1:
                    res[i][j] = self.right_bound_inner(grid, i,  j)
                    break
                res[i][j] = self.puasson(grid, i, j)
        # print(res)
        return res


class A_der:

    def __init__(self, Ny):
        self.N_x = n * p
        self.Ny = Ny

        self.hx = Lx / (n * (p - 1))  # Шаг сетки по x
        self.hy = Ly / (Ny - 1)  # Шаг сетки по y

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
        second = 1/(2*np.power(self.hy, 2))*(u_der[0][j-1] + u_der[0][j+1])
        return first - second - u_der[1][j]/np.power(self.hx, 2)

    def right_bound_exter(self, u, u_der, i, j):
        first = (1 / np.power(self.hy, 2) + 1 / np.power(self.hx, 2) + h_der(u[i][j]) / self.hx / lamda) * u_der[i][j]
        second = 1 / (2 * np.power(self.hy, 2)) * (u_der[i][j - 1] + u_der[i][j + 1])
        third = u_der[i - 1][j] / np.power(self.hx, 2)
        return first - second - third

    def left_bound_inner(self, u, u_der, i, j):
        first = (1 / np.power(self.hy, 2) + 1 / np.power(self.hx, 2) + h_der(u[i][j]) / self.hx / lamda) * u_der[i][j]
        second = 1 / (2 * np.power(self.hy, 2)) * (u_der[i][j - 1] + u_der[i][j + 1])
        third = h_der(u[i + 1][j]) / (self.hx * lamda) * u_der[i + 1][j]
        return first - second - third - u_der[i - 1][j] / np.power(self.hx, 2)

    def right_bound_inner(self, u, u_der, i, j):
        first = (1 / np.power(self.hy, 2) + 1 / np.power(self.hx, 2) + h_der(u[i][j]) / self.hx / lamda) * u_der[i][j]
        second = 1 / (2 * np.power(self.hy, 2)) * (u_der[i][j - 1] + u_der[i][j + 1])
        third = h_der(u[i - 1][j]) / (self.hx * lamda) * u_der[i - 1][j]
        return first - second - third - u_der[i + 1][j] / np.power(self.hx, 2)

    def apply_operator(self, u, u_der):
        # res = np.zeros((self.N_x, self.Ny))
        res = u_der.copy()
        for j in range(1, self.Ny - 1):
            res[0][j] = self.left_bound_exter(u, u_der, j)
            res[self.N_x - 1][j] = self.right_bound_exter(u, u_der, self.N_x - 1, j)

        for i in range(1, self.N_x - 1):
            for j in range(1, self.Ny - 1):
                if i % p == 0:
                    res[i][j] = self.left_bound_inner(u, u_der, i, j)
                    break
                if i % p == p - 1:
                    res[i][j] = self.right_bound_inner(u, u_der, i, j)
                    break
                res[i][j] = self.puasson(u_der, i, j)
        return res
        # print(res)

# print(grid)
# S = A_der(hx, hy)
# S.apply_operator()
# S = A(hx, hy)
# S.apply_operator()