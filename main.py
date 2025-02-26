from solver import Solver
from solver_fsolve import solve
from A_der import A, A_der
import numpy as np

s = Solver()
# print(s.solve())
s.solve()
# s.init(s.p)
# print(s.A.apply_operator(np.zeros_like(s.p)))
#u_in = solve()
u_in = np.zeros((10, 10))
s.plot_heatmap(u_in)
#A_ = A()
#A_.plot_operator_application()
# A_d = A_der()
# A_d.plot_operator_application()