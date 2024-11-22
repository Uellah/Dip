import numpy as np
from utils import out_to_file

OUT_FILE = out_to_file('rs.csv')

sigma_0 = 5.67e-8
u_l = 100
u_r = 100
lamda = 100

def h(u):
    return sigma_0 * np.abs(np.power(u, 3))*u

def h_der(u):
    return 4 * sigma_0 * np.power(u, 3)

n = 2   # количество пластин.
p = 3   # количество точек по x в одной пластине.
N_y = 3   # количество точек по y.
N_x = n * p    # количество точек по x
Lx = 1
Ly =  1
hx = Lx / (n * (p - 1))  # Шаг сетки по x
hy = Ly / (N_y - 1)  # Шаг сетки по y
grid = np.ones((N_y, N_x))  # Общая сетка всех пластин

OUT_FILE.write_numpy_to_csv(grid)

print(grid)
class A:
    def __init__(self, hx, hy):
        self.hx = hx
        self.hy = hy

    def second_x_der(self, u, j, i):
        res = (u[i][j-1] - 2 * u[i][j] + u[i][j-1])/np.power(self.hy, 2)
        return res

    def second_y_der(self, u, j, i):
        res = (u[i-1][j] - 2 * u[i][j] + u[i+1][j])/np.power(self.hx, 2)
        return res

    def puasson(self, u, j, i):
        return self.second_x_der(u, j, i) + self.second_y_der(u, i, j)

    def left_bound_exter(self, u, j):
        first = (u[j][0] - u[j][1])/np.power(self.hx, 2)
        second = self.second_y_der(u, 0, j)/2
        third = (h(u_l) - h(u[j][0]))/self.hx/lamda
        return first - second - third

    def right_bound_exter(self, u, j, i):
        first = (u[i][j] - u[i-1][j])/np.power(self.hx, 2)
        second = self.second_y_der(u, j, i)/2
        third = (h(u_r) - h(u[i][j]))/self.hx/lamda
        return first - second - third

    def left_bound_inner(self, u, j, i):
        first = (u[i][j] - u[i - 1][j]) / np.power(self.hx, 2)
        second = self.second_y_der(u, j, i) / 2
        third = (h(u[i+1][j])- h(u[i][j])) / self.hx / lamda
        return first - second - third

    def right_bound_inner(self, u, j, i):
        first = (u[i][j] - u[i + 1][j]) / np.power(self.hx, 2)
        second = self.second_y_der(u, j, i) / 2
        third = (h(u[i-1][j]) - h(u[i][j])) / self.hx / lamda
        return first - second - third

    def apply_operator(self):
        res = np.zeros((N_y, N_x))
        print(N_x, N_y)
        for i in range(N_x):
            res[0][i] = grid[0][i]
            res[N_y-1][i] = grid[N_y-1][i]

        for i in range(1, N_y-1):
            res[i][0] = self.left_bound_exter(grid, i)
            res[i][N_x-1] = self.right_bound_exter(grid, N_x-1, i)

        print(res)
        print(h(u_l) - h(1)/lamda/hx)
        # for i in range(1, N_x - 1):
        #     if i
        #     for j in range(1, N_y - 1):



class A_der:
    def __init__(self, hx, hy):
        self.hx = hx
        self.hy = hy

    def second_y_der(self, u, i, j):
        res = (u[i][j - 1] - 2 * u[i][j] + u[i][j - 1]) / np.power(self.hy, 2)
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

A_ = A(hx, hy)
res = A_.apply_operator()
print(res)