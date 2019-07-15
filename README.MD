# A new fully distributed triangulation in deal.II

This repo contains programs clarifying the usage of the new `parallel::fullydistributed::Triangulation` (short: `PFT`)
in deal.II. For implementation details see the attached presentation and/or the source code.

In the following, I give some short examples using `PFT`. The exaples are ordered according to their difficulty in 
steps.

## Step 1: Static, globally uniformly refined mesh

Convert the fine mesh of a `parallel::distributed::Triangulation` (`PDT`) to a `PFT`.

```cpp
mkdir direct-setup
cd direct-setup
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 8
```

## Step 2: Static, globally non-uniformly refined mesh 

???

## Step 3: Static, globally uniformly refined mesh with multigrid levels

Convert the fine mesh of a `PDT` with all multigrid levels to a `PFT`.

## Step 4: Partition a serial mesh 

using METIS

## Step 5: Repartition a distributed mesh

 using ParMETIS


