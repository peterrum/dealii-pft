make -j10

cd step-0
mpirun -np 5 ./step-0 2 4
cd ..


cd step-1
mpirun -np 5 ./step-1 2 8
cd ..



cd step-3 
mpirun -np 5 ./step-3 2 4 8
cd ..



cd step-4
mpirun -np 5 ./step-4 2 4 8
cd ..



cd step-5
mpirun -np 5 ./step-5 2 4 8
cd ..



cd step-6 
mpirun -np 5 ./step-6 2 4 8
cd ..