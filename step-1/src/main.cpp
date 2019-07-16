#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "../../common/utilities.h"

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;



template<int dim>
void
test(int n_refinements, MPI_Comm comm)
{
  // create pdt
  parallel::distributed::Triangulation<dim> tria_pdt(
    comm,
    dealii::Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
  GridGenerator::hyper_cube(tria_pdt);
  tria_pdt.refine_global(n_refinements);

  // extract relevant information from pdt to be able to create pft
  std::vector<CellData<dim>>         cells;
  std::vector<Point<dim>>            vertices;
  std::vector<int>                   boundary_ids;
  std::map<int, std::pair<int, int>> coarse_lid_to_gid;
  std::vector<Part>                  parts;

  parallel::fullydistributed::Utilities::copy_from_distributed_triangulation(
    tria_pdt, cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // create pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  tria_pft.reinit(cells, vertices, boundary_ids, coarse_lid_to_gid, parts);

  // output meshes as VTU
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(tria_pdt, "trid_pdt", true, true);
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "trid_pft", true, true);
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 2, ExcMessage("You have not provided two command-line arguments."));

  const int dim           = atoi(argv[1]);
  const int n_refinements = atoi(argv[2]);

  if(dim == 2)
    test<2>(n_refinements, comm);
  else if(dim == 3)
    test<3>(n_refinements, comm);
  else
    AssertThrow(false, ExcMessage("Only working for dimensions 2 and 3!"));
}