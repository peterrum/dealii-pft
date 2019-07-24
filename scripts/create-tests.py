dim_min = 2;
dim_max = 3;

procs_min = 1;
procs_max = 20;


def n_cell_hyper_l(d, r):
    return (2 ** d) ** r * (2 ** d - 1)

def n_cell_hyper_shell(d, r, s):
    return (2 ** d) ** r * s

def n_cell_subdivided_hyper_cube(d, r, s):
    return (2 ** d) ** r * s**d




def print_stopper():
    print "if [ -f STOP ]; then"                                                                                               
    print "  return"                                                                                                             
    print "fi" 

def step_0():
    for d in range(dim_min, dim_max+1):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                cells = n_cell_hyper_l(d, r)
                if(p <= cells and cells <= 100000):
                    print_stopper()
                    print "mpirun -np %d ../step-0/step-0 %d %d" % (p, d, r)
                    
def step_1():
    for d in range(dim_min, dim_max+1):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                cells = (2 ** d) ** r
                if(cells <= 10000*p):
                    print_stopper()
                    print "mpirun -np %d ../step-1/step-1 %d %d" % (p, d, r)
                    
def step_2():
    for d in range(2, 3):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                for s in range(4, 11):
                    cells = n_cell_hyper_shell(d, r, s)
                    if(cells <= 100000):
                        print_stopper()
                        print "mpirun -np %d ../step-2/step-2 %d %d %d" % (p, d, r, s)
                    
def step_3():
    for d in range(2, 3):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                for s in range(1, 11):
                    if(n_cell_subdivided_hyper_cube(d, r, s) <= 20000):
                        print_stopper()
                        print "mpirun -np %d ../step-3/step-3 %d %d %d" % (p, d, r, s)
                    
def step_4():
    for d in range(2, 3):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                for s in range(1, 11):
                    if(n_cell_subdivided_hyper_cube(d, r, s) <= 10000*p):
                        print_stopper()
                        print "mpirun -np %d ../step-4/step-4 %d %d %d" % (p, d, r, s)
                    
def step_5():
    for d in range(2, 3):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                for s in range(1, 11):
                    if(n_cell_subdivided_hyper_cube(d, r, s) <= 20000):
                        print_stopper()
                        print "mpirun -np %d ../step-5/step-5 %d %d %d" % (p, d, r, s)
                    
def step_6():
    for d in range(2, 3):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                for s in range(1, 11):
                    if(n_cell_subdivided_hyper_cube(d, r, s) <= 20000):
                        print_stopper()
                        print "mpirun -np %d ../step-6/step-6 %d %d %d" % (p, d, r, s)
                        
                        

def step_7():
    for d in range(1, 3+1):
        for p in range(procs_min, procs_max+1):
            for r in range(0, 7):
                if(n_cell_hyper_l(d, r) <= 1000):
                    print_stopper()
                    print "mpirun -np %d ../step-7/step-7 %d %d" % (p, d, r)

def main():
    step_0()
    step_1()
    step_2()
    step_3()
    step_4()
    step_5()
    step_6()
    step_7()
        
  
if __name__ == "__main__":
    main()