make -j10

# fine grid of serial triangulation
cd step-0
mpirun -np 5 ./step-0 2 4
cd ..



# fine grid of distributed triangulation
cd step-1
mpirun -np 5 ./step-1 2 8
cd ..



# multigrid levels of distributed triangulation (+hanging node)
cd step-2 
mpirun -np 7 ./step-2 2 4 10
cd ..



# multigrid levels of serial triangulation
cd step-3 
mpirun -np 5 ./step-3 2 4 8
cd ..



# multigrid levels of distributed triangulation
cd step-4
mpirun -np 5 ./step-4 2 4 8
cd ..


# partitioning of serial mesh 
#cd step-5
#mpirun -np 5 ./step-5 2 4 8
#cd ..



# partitioning of serial mesh with periodic bcs
#cd step-6 
#mpirun -np 5 ./step-6 2 4 8
#cd ..
