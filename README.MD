# A new fully distributed triangulation in deal.II

This repo contains programs clarifying the usage of the new `parallel::fullydistributed::Triangulation` (short: `PFT`)
in deal.II. For implementation details see the attached presentation (WIP) and/or the source code.

**Note:** The origin of the name `parallel::fullydistributed::Triangulation` is the need to 
distinguish the class from `parallel::distributed::Triangulation` and to emphasis that also
the coarse mesh is partitioned. Please feel free to suggest alternative names!  

## Requirements

An (incomplete) list of requirements:
- [x] extract from `dealii::Triangulation` (serial triangulation) 
- [x] extract from `dealii::parallel::distributed::Triangulation` (parallel triangulation) 
- [x] static mesh
- [x] hanging nodes
- [x] geometric multigrid
- [x] periodicity
- [x] 1D/2D/3D
- [x] I/O from/to `*.pft`-files
- [ ] adaptive mesh

The ticks indicate that this feature has been implemented.

## Concept

The motivation to implement the new `PDT` framework origins in the following observations for 
complex triangulations and/or for given meshed created by an external mesher. W
e regard complex geometries which can be meshed only with a non-negligible 
amount of coarse cell (>10.000):

- storing the coarse-grid information on every process is too expensive from a  memory point of view
  (especially, if you consider that a single compute node on SuperMUC-NG has 48 CPUs, which would 
   mean that the coarse grid is duplicated 48 on the same node in the case that a pure MPI program
   is considered). Normally, a process only needs a small section of the global triagnulation levels,
   i.e. a small section of the coarse grid, such that a **partitioning of the coarse grid** is
   not negligible.

- the distribution of the active cells - on the finest level - among all processes by simply 
  partitioning a space-filling curve might not lead to an optimal result: e.g. partitions belonging
  to the same process might be discontiguous, leading to increased communication amount (within a
  node and islands). **Graph-based partitioning** algorithms might be sound alternative.

**Note:** The first aspect could be tackled via applying hybrid programming (MPI+X), e.i, all threads of
a processes access the same triangulation class. This implementation does not investigate this 
aspect but rather investigates partitioning algorithms which explicitly exploit
hardware properties regarding NUMA-domain, node and island size.

To be able to construct a fully partitioned triangulation, which fulfills the [requirements](), we need
following ingredients:
1. a locally relevant coarse-grid triangulation (vertices, cells with material and manifold ID, boundary IDs)
2. a mapping of the locally relevant coarse-grid triangulation into the global coarse-grid triangulation
3. information about which cell should be refined and information regarding the sudomain id and the level 
  subdomain id of each cell.

This set of information is bundled in `parallel::fullydistributed::ConstructionData`, such that the user has 
to fill this data structure - in a pre-processing step - before actually creating the triangulation.
As you will see, we provide a front-end to serialize and deserialize triangulations and to convert
serial and distributed meshes.

## Additional information

- A triangulation does not come alone. It is accompanied with a DoFHanlder policy. The new 
policy works similarly as the policy of `dealii::parallel::distributed::Triangulation` except
that the local enumeration of the coarse cells have to translated to a global enumeration before
sending data, and vice versa if the received data should be processed.

- The new triangulation inherits from `parallel::Triangulation`. Small modifications were needed
to enable parallel 1D simulations and periodicity (which is normally imposed on the coarse-grid)/

## Examples (and nice pictures)

The following pictures show use cases for which the new `PFT` framework has been applied.

### NACA 0012 airfoil

NACA 0012 airfoil with 156 coarse grid cells and 2 refinement levels. The fully 
distributed triangulation was created as described in step-6.

![naca](figures/naca.png)

Thanks to: [Elias Dejene](https://github.com/eliasstudiert)

### Lung

Deformed mesh of a lung with six generations with different refinement levels:

![lung](figures/lung_generations.png)

Thanks to: [Martin Kronbichler](https://github.com/kronbichler)

## Tutorials

In the following, I give some short examples using `PFT`. The examples are in contrast to the 
examples in the last section quite academic and are ordered according to their difficulty in 
steps.

The structure of each tutorial is as follows:


The flexibility of having a front-end and a back-end should become obvious.

**Note:** The tutorials are a list of tested applications. Please feel free to 
suggests additional use cases and/or to extend the relevant classes.


















### Step 1: Static, globally uniformly refined mesh (serial)

**Short description:** Convert a colored (partitioned) serial fine mesh of a type`dealii::Triangulation` to a `PFT`.

We create a serial triangulation `dealii::Triangulation`. We partition the active cells
by simply applying `dealii::GridTools::partition_triangulation()` onto it. 

The locally relevant mesh information is extracted by calling the new function:
```cpp
template<int dim, int spacedim = dim>
ConstructionData<dim, spacedim>
copy_from_triangulation(const dealii::Triangulation<dim, spacedim> & tria,
                        const Triangulation<dim, spacedim> &         tria_pft)
```
and used to setup `PFT`:
```
template<int dim, int spacedim = dim>
reinit(ConstructionData<dim, spacedim> data);
```

**Execution:**

```bash
cd step-1
make
mpirun -np 5 ./step-1 2 4 # dim, n_refinements
```

**Results:**

![step-1-overview](step-1/pictures/overview.png)

**Note:** This use case is motivated by the tests of `parallel::split::Triangulation`, which 
have been discussed
in [PR #3956](https://github.com/dealii/dealii/pull/3956).












### Step 2: (De)serialization

**Short description:** This steps presents the serialization and deserialization capabilities of `pft`.

As in [step-0](https://github.com/peterrum/dealii-pft#step-0-static-globally-uniformly-refined-mesh-serial)
a serial mesh is converted into `pft`. 
However, in between the construction data is written to files with the help of
`boost`:
```cpp
parallel::fullydistributed::Utilities::serialize(construction_data, file_name, comm);
```
 and read from right away from them:
```cpp
auto construction_data = parallel::fullydistributed::Utilities::deserialize<dim>(file_name, comm);
```

**Execution:**

```bash
cd step-2
make
mpirun -np 5 ./step-2 2 4 # dim, n_refinements
```

















### Step 3: Static, globally uniformly refined mesh with multigrid levels (serial)

**Short description:** Convert the fine mesh of a serial mesh with all multigrid levels to a `PFT`.

**Execution:**

```bash
cd step-3 
make
mpirun -np 5 ./step-3 2 4 8
```

**Results:**

![step-3-overview](step-3/pictures/overview.png)



















### Step 4: Static, globally non-uniformly refined mesh 

**Short description:** 

**Execution:**

```bash
cd step-4
make
mpirun -np 5 ./step-4 2 4 8
```


**Results:**

![step-4-overview](step-4/pictures/overview.png)











### Step 5: Partition a serial mesh with periodic faces

**Short description:** The same as [step-5](https://github.com/peterrum/dealii-pft#step-5-partition-a-serial-mesh), but with periodic faces in x-direction.


**Execution:**

```bash
cd step-5 
make
mpirun -np 5 ./step-5 2 4 8
```

**Results:**

![step-5-overview](step-5/pictures/overview.png)
















### Step 6: Multi-level partitioning of a serial mesh 

**Short description:** Create a serial triangulation on selected processes, partition the mesh 
using METIS, taking in account node-locality, and finally distribute mesh.

For distribution the construction data a serialized to a buffer (see [step-2](https://github.com/peterrum/dealii-pft#step-2-deserialization)), 
which is sent via `MPI`.


**Execution:**

```bash
cd step-6
make
mpirun -np 5 ./step-6 2 4 8
```



















### Step 7: Static, globally uniformly refined mesh (distributed)

**Short description:** Convert the fine mesh of a `parallel::distributed::Triangulation` (`PDT`).


**Execution:**

```bash
cd step-7
make
mpirun -np 5 ./step-7 2 8
```












### Step 8: Static, globally uniformly refined mesh with multigrid levels (distributed)

**Short description:** Convert the all mesh levels of a `PDT`, needed for multigrid.


**Execution:**

```bash
cd step-8
make
mpirun -np 8 ./step-8 4 3 5
```

**Results:**

![step-8-overview](step-8/pictures/overview.png)











