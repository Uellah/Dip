from sympy import symbols, Function, Eq, Matrix, diff, latex, pprint
from sympy.abc import i, j


# Определение параметров
hx, hy, lam, sigma_0 = symbols('h_x h_y lam sigma_0')
p = symbols('p', integer=True)  # Число узлов на каждой пластине
n = symbols('n', integer=True)  # Число пластин
N = symbols('N', integer=True)  # Число разбиений по оси y
N = 3
n = 1
p = 2
# Функция h(u) = sigma_0 * |u| * u^3
def h(u):
    return sigma_0 * u ** 4


# Функция для генерации символьной переменной u_i_j
def u(i, j):
    return symbols(f'u_{i}_{j}')


# Генерация уравнения для внутренних узлов
def poisson_equation(i, j):
    return Eq(
        - (u(i + 1, j) - 2 * u(i, j) + u(i - 1, j)) / hx ** 2
        - (u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)) / hy ** 2
        - symbols(f'f_{i}_{j}') / lam,
        0
    )


# Граничные условия на левой и правой внешних границах
def boundary_equation_left(j):
    return Eq(
        - (u(1, j) - u(0, j)) / hx ** 2
        - (u(0, j + 1) - 2 * u(0, j) + u(0, j - 1)) / (2 * hy ** 2)
        - (h(symbols('u_l')) - h(u(0, j))) / (lam * hx)
        - symbols(f'f_{0}_{j}') / (2 * lam),
        0
    )


def boundary_equation_right(j):
    i = n*(p+1) - 1
    return Eq(
        - (u(i - 1, j) - u(i, j)) / hx ** 2
        - (u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)) / (2 * hy ** 2)
        - (h(symbols('u_r')) - h(u(i, j))) / (lam * hx)
        - symbols(f'f_{i}_{j}') / (2 * lam),
        0
    )


# Левая и правая внутренняя границы между пластинами
def boundary_equation_left_inner(i, j):
    return Eq(
        - (u(i - 1, j) - u(i, j)) / hx ** 2
        - (u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)) / (2 * hy ** 2)
        - (h(u(i + 1, j)) - h(u(i, j))) / (lam * hx)
        - symbols(f'f_{i}_{j}') / (2 * lam),
        0
    )


def boundary_equation_right_inner(i, j):
    return Eq(
        - (u(i + 1, j) - u(i, j)) / hx ** 2
        - (u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)) / (2 * hy ** 2)
        - (h(u(i - 1, j)) - h(u(i, j))) / (lam * hx)
        - symbols(f'f_{i}_{j}') / (2 * lam),
        0
    )


# Функция для создания системы уравнений на сетке
def create_system():
    equations = []

    # Внутренние узлы
    for i1 in range(0, n):
        for i2 in range(1, p):
            for j in range(1, N):
                equations.append(poisson_equation(i1*(p+1) + i2, j))

    #Левые и правые внешние границы
    for j in range(1, N):
        equations.append(boundary_equation_left(j))  # левая внешняя граница
        equations.append(boundary_equation_right(j))  # правая внешняя граница

    for i1 in range(0, n-1):
        for j in range(1, N):
            equations.append(boundary_equation_left_inner(i1*(p+1)+p, j))

    for i1 in range(0, n - 1):
        for j in range(1, N):
            equations.append(boundary_equation_right_inner((i1+1) * (p + 1) + p, j))

    return equations


# Функция для вычисления якобиана
def compute_jacobian(equations):
    # Определяем все переменные
    variables = [u(i, j) for i in range(0, n*(p+1)) for j in range(1, N)]

    # Преобразуем систему уравнений в матричную форму
    lhs_expressions = [eq.lhs for eq in equations]

    # Вычисляем якобиан
    jacobian_matrix = Matrix(lhs_expressions).jacobian(variables)
    return jacobian_matrix


# # Пример использования
# system_eqs = create_system()
# jacobian = compute_jacobian(system_eqs)
#
# # Вывод системы уравнений
# print("Система уравнений:")
# for eq in system_eqs:
#     pprint(eq)
#
# # Вывод якобиана в формате LaTeX
# print("\nЯкобиан системы в LaTeX:")
# latex_jacobian = latex(jacobian)

