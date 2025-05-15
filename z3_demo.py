# -*- coding: utf-8 -*-
from z3 import *

def solution():
    z3._main_ctx = Context()
    solver = Solver()
    evan_dollars = Real('evan_dollars')
    markese_dollars = Real('markese_dollars')
    solver.add(markese_dollars == evan_dollars - 5)
    total_dollars = Real('total_dollars')
    solver.add(total_dollars == evan_dollars + markese_dollars)
    solver.add(total_dollars == 37)
    result = Real('result')
    solver.add(result == markese_dollars)
    if solver.check() == unsat:
        return False, "UNSAT"
    final_val = solver.model().eval(result)
    solver.add(result != final_val)
    if solver.check() == sat:
        return False, "AMBIG"
    return True, final_val

if __name__ == '__main__':
    solution()
