#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include "../../common/utilities.h"

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;



template<int dim>
void
test(int n_refinements, MPI_Comm comm)
{
  // create pdt
  Triangulation<dim> basetria;
  GridGenerator::hyper_L(basetria);
  basetria.refine_global(n_refinements);

  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     basetria,
                                     SparsityTools::Partitioner::metis);
  GridTools::partition_multigrid_levels(basetria);

  // extract relevant information from pdt to be able to create pft
  std::vector<CellData<dim>>         cells;
  std::vector<Point<dim>>            vertices;
  std::vector<int>                   boundary_ids;
  std::map<int, std::pair<int, int>> coarse_lid_to_gid;
  std::vector<Part>                  parts;

  parallel::fullydistributed::Utilities::copy_from_serial_triangulation(
    basetria, comm, cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // create pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.reinit(cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // output meshes as VTU
  GridOut grid_out;
  if(Utilities::MPI::this_mpi_process(comm) == 0)
    grid_out.write_mesh_per_processor_as_vtu(basetria, "tria", true, true);
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "tria_pft", true, true);
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 2, ExcMessage("You have not provided two command-line arguments."));

  const int dim           = atoi(argv[1]);
  const int n_refinements = atoi(argv[2]);

  if(dim == 1)
    test<1>(n_refinements, comm);
  else if(dim == 2)
    test<2>(n_refinements, comm);
  else if(dim == 3)
    test<3>(n_refinements, comm);
  else
    AssertThrow(false, ExcMessage("Only working for dimensions 1, 2, and 3!"));
}