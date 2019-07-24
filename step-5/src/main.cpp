#include <deal.II/base/geometry_info.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>

#include "../../common/utilities.h"

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;


template<int dim, int spacedim = dim>
void
test(const int n_refinements, const int n_subdivisions, MPI_Comm comm)
{
  // create pft
  parallel::fullydistributed::Triangulation<dim, spacedim> tria_pft(
    comm, parallel::fullydistributed::Triangulation<dim>::construct_multigrid_hierarchy);

  GridTools::AdditionalData additional_data;
  additional_data.partition_group = GridTools::PartitioningGroup::shared;

  // create serial triangulation and extract relevant information
  auto construction_data =
    parallel::fullydistributed::Utilities::create_and_partition<dim, spacedim>(
      [&](dealii::Triangulation<dim, spacedim> & tria) mutable {
        GridGenerator::subdivided_hyper_cube(tria, n_subdivisions);
        tria.refine_global(n_refinements);
      },
      tria_pft,
      additional_data);

  // actually create triangulation
  tria_pft.reinit(construction_data);

  // output mesh as VTU
  GridOut grid_out;
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

  ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0);

  try
  {
    // clang-format off
    pcout << "Run step-5: "
          << " p=" << std::setw(2) << dealii::Utilities::MPI::n_mpi_processes(comm)
          << " d=" << std::setw(2) << dim
          << " r=" << std::setw(2) << n_refinements
          << " s=" << std::setw(2) << n_subdivisions
          << ":";
    // clang-format on

    if(dim == 1)
      test<1>(n_refinements, n_subdivisions, comm);
    else if(dim == 2)
      test<2>(n_refinements, n_subdivisions, comm);
    else if(dim == 3)
      test<3>(n_refinements, n_subdivisions, comm);
    else
      AssertThrow(false, ExcMessage("Only working for dimensions 1, 2, and 3!"));
    pcout << " success...." << std::endl;
  }
  catch(...)
  {
    pcout << " failed...." << std::endl;
  }
}