from sympy import symbols, Function, diff, Matrix, latex
from utils import out_to_file

OUT_FILE = out_to_file('der_A.txt')

# Определяем переменные
i, j, h_x, h_y, sigma, lambda_ = symbols('i j h_x h_y sigma lambda')
u = Function('u')


# Определяем функцию h(u) = sigma * u^4
def h(u_var):
    return sigma * u_var ** 4


def out_to_f(expr, arr_of_var):
    derivatives = [diff(expr, var) for var in arr_of_var]

    # Создаем вектор строку из частных производных
    gradient_vector = Matrix(1, len(derivatives), derivatives)

    variable_column = Matrix(len(arr_of_var), 1, arr_of_var)

    # Выполним умножение вектора строки на вектор столбец
    result = gradient_vector * variable_column

    # Выводим результат в файл и возвращаем его
    OUT_FILE.print_string_to_file(latex(result))

    return result


def get_der_puasson():
    # Задаем выражение с использованием u(i+1, j), u(i, j), u(i-1, j)
    expression = (u(i + 1, j) - 2 * u(i, j) + u(i - 1, j)) / h_x ** 2 + (
                u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)) / h_y ** 2

    OUT_FILE.print_string_to_file(latex(expression))

    # Список переменных для дифференцирования
    variables = [u(i + 1, j), u(i, j), u(i - 1, j), u(i, j + 1), u(i, j), u(i, j - 1)]

    # return out_to_f(expression, variables)


def get_der_left_external():
    # Определяем переменные для выражения
    u_1j = u(1, j)  # u_{1, j}
    u_0j = u(0, j)  # u_{0, j}
    u_0j1 = u(0, j + 1)  # u_{0, j+1}
    u_0j_1 = u(0, j - 1)  # u_{0, j-1}
    u_l = symbols('u_l')  # u_l для h(u_l)

    # Задаем выражение
    expression = (
            -(u_1j - u_0j) / h_x ** 2
            - (u_0j1 - 2 * u_0j + u_0j_1) / (2 * h_y ** 2)
            - (h(u_l) - h(u_0j)) / (lambda_ * h_x)
    )

    OUT_FILE.print_string_to_file(latex(expression))

    # Переменные для дифференцирования
    variables = [u_1j, u_0j, u_0j1, u_0j_1]

    # return out_to_f(expression, variables)


def get_der_right_external():
    # Определяем переменные для выражения
    u_i_1j = u(i - 1, j)  # u_{i-1, j}
    u_ij = u(i, j)  # u_{i, j}
    u_ij1 = u(i, j + 1)  # u_{i, j+1}
    u_ij_1 = u(i, j - 1)  # u_{0, j-1}
    u_r = symbols('u_r')  # u_l для h(u_l)

    # Задаем выражение
    expression = (
            -(u_i_1j - u_ij) / h_x ** 2
            - (u_ij1 - 2 * u_ij + u_ij_1) / (2 * h_y ** 2)
            - (h(u_r) - h(u_ij)) / (lambda_ * h_x)
    )

    OUT_FILE.print_string_to_file(latex(expression))

    # Переменные для дифференцирования
    variables = [u_i_1j, u_ij, u_ij1, u_ij_1]

    # return out_to_f(expression, variables)


def get_der_left_inner():
    # Определяем переменные для выражения
    u_i_1j = u(i - 1, j)  # u_{i-1, j}
    u_ij = u(i, j)  # u_{i, j}
    u_ij1 = u(i, j + 1)  # u_{i, j+1}
    u_ij_1 = u(i, j - 1)  # u_{0, j-1}
    u_i1_j = u(i + 1, j)  # u_{i+1, j}
    # Задаем выражение
    expression = (
            -(u_i_1j - u_ij) / h_x ** 2
            - (u_ij1 - 2 * u_ij + u_ij_1) / (2 * h_y ** 2)
            - (h(u_i1_j) - h(u_ij)) / (lambda_ * h_x)
    )

    OUT_FILE.print_string_to_file(latex(expression))

    # Переменные для дифференцирования
    variables = [u_i_1j, u_ij, u_ij1, u_ij_1, u_i1_j]

    # return out_to_f(expression, variables)


def get_der_right_inner():
    # Определяем переменные для выражения
    u_i_1j = u(i - 1, j)  # u_{i-1, j}
    u_ij = u(i, j)  # u_{i, j}
    u_ij1 = u(i, j + 1)  # u_{i, j+1}
    u_ij_1 = u(i, j - 1)  # u_{0, j-1}
    u_i1_j = u(i + 1, j)  # u_{i+1, j}
    # Задаем выражение
    expression = (
            -(u_i1_j - u_ij) / h_x ** 2
            - (u_ij1 - 2 * u_ij + u_ij_1) / (2 * h_y ** 2)
            - (h(u_i_1j) - h(u_ij)) / (lambda_ * h_x)
    )

    OUT_FILE.print_string_to_file(latex(expression))

    # Переменные для дифференцирования
    variables = [u_i_1j, u_ij, u_ij1, u_ij_1, u_i1_j]

    # return out_to_f(expression, variables)


# Вызываем функцию и печатаем результат
OUT_FILE.clear_file()
get_der_puasson()
get_der_left_external()
get_der_right_external()
get_der_left_inner()
get_der_right_inner()
OUT_FILE.print_string_to_file('\n')
