# A new fully distributed triangulation in deal.II

This repo contains programs clarifying the usage of the new `parallel::fullydistributed::Triangulation` (short: `PFT`)
in deal.II. For implementation details see the attached presentation and/or the source code.

**Note:** The origin of the name `parallel::fullydistributed::Triangulation` is the need to 
distinguish the class from `parallel::distributed::Triangulation` and to emphasis that also
the coarse mesh is partitioned. Please feel free to suggest alternative names!  

## Requirements

An (incomplete) list of requirements:
- [x] extract from `dealii::Triangulation` (serial triangulation) 
- [x] extract from `dealii::distributed::Triangulation` (parallel triangulation) 
- [x] static mesh
- [ ] hanging nodes
- [ ] adaptive mesh
- [x] geometric multigrid
- [x] periodicity
- [x] 1D/2D/3D

## Concept

## Examples (and nice pictures)

### NACA 0012 airfoil

Thanks to: [Elias Dejene](https://github.com/eliasstudiert)

### Can (moving mesh)

Thanks to: [Daniel Dengler](https://github.com/daniel-dengler)

### Lung

Thanks to: [Martin Kronbichler](https://github.com/kronbichler)

## Tutorials

In the following, I give some short examples using `PFT`. The examples are ordered according to their difficulty in 
steps.

The structure of each tutorial is as follows:


The flexibility of having a front-end and a back-end should become obvious.

**Note:** The tutorials are a list of tested applications. Please feel free to 
suggests additional use cases and/or to extend the relevant classes.

### Step 0: Static, globally uniformly refined mesh (serial)

**Short description:** Convert a colored serial fine mesh of a type`dealii::Triangulation` to a `PFT`.

In the first step, all processes color all active cells of a serial triangulation in the same way.
For this we simply apply `dealii::GridTools::partition_triangulation()` onto
the serial triangulation. In a second step, the locally relevant mesh info is extracted
and used to setup `PFT`.

**Execution:**

```bash
cd step-0
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 4 # dim, n_refinements
```

**Results**

![step-0-overview](step-0/pictures/overview.png)

**Note:** This use case is motivated by the tests of `parallel::split::Triangulation`, which 
have been discussed
in [PR #3956](https://github.com/dealii/dealii/pull/3956).

### Step 1: Static, globally uniformly refined mesh (parallel)

Convert the fine mesh of a `parallel::distributed::Triangulation` (`PDT`) to a `PFT`.

```bash
cd step-1
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 8
```

### Step 2: Static, globally non-uniformly refined mesh 

???

### Step 3: Static, globally uniformly refined mesh with multigrid levels (serial)

Convert the fine mesh of a serial mesh with all multigrid levels to a `PFT`.


```bash
cd step-3 
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 4 8
```

### Step 4: Static, globally uniformly refined mesh with multigrid levels (distributed)

Convert the fine mesh of a `PDT` with all multigrid levels to a `PFT`.


```bash
cd step-3 
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 4 8
```

### Step 5: Partition a serial mesh 

using METIS

```bash
cd step-4
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 4 8
```

### Step 6: Partition a serial mesh with periodic faces

The same as step-4, but with periodic faces in x-direction.

```bash
cd step-5 
mkdir build
cd build
cmake ..
mpirun -np 5 ./main 2 4 8
```

### Step 7: Repartition a distributed mesh

 using ParMETIS

### Step 8: Advanced topics

- two-level partitioning to enable data locality on a compute node


