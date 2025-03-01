from solver import Solver
from solver_fsolve import solve
from A_der import A, A_der
import numpy as np

s = Solver()
an = s.get_analytical_solve()
s.solve()
#u_in = solve()
print(s.get_r_norm(s.p, an))
s.plot_heatmap(an)
#A_ = A()
#A_.plot_operator_application()