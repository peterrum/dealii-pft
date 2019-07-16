#include <deal.II/base/geometry_info.h>
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
test(int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  // create pdt
  Triangulation<dim> basetria(Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::subdivided_hyper_cube(basetria, n_subdivisions);
  basetria.refine_global(n_refinements);

  GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                     basetria,
                                     SparsityTools::Partitioner::metis);
  GridTools::partition_multigrid_levels(basetria);

  // create instance of pft
  parallel::fullydistributed::Triangulation<dim> tria_pft(
    comm, parallel::fullydistributed::Triangulation<dim>::construct_multigrid_hierarchy);

  // extract relevant information form serial triangulation
  auto construction_data =
    parallel::fullydistributed::Utilities::copy_from_serial_triangulation(basetria, tria_pft);

  // actually create triangulation
  tria_pft.reinit(construction_data);

  // output meshes as VTU
  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(basetria, "trid_pdt", true, true);
  grid_out.write_mesh_per_processor_as_vtu(tria_pft, "trid_pft", true, true);
}



int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  AssertThrow(argc > 3, ExcMessage("You have not provided two command-line arguments."));

  const int dim            = atoi(argv[1]);
  const int n_refinements  = atoi(argv[2]);
  const int n_subdivisions = atoi(argv[3]);

  if(dim == 1)
    test<1>(n_refinements, n_subdivisions, comm);
  else if(dim == 2)
    test<2>(n_refinements, n_subdivisions, comm);
  else if(dim == 3)
    test<3>(n_refinements, n_subdivisions, comm);
  else
    AssertThrow(false, ExcMessage("Only working for dimensions 2 and 3!"));
}