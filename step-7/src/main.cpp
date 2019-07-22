#include <deal.II/base/mpi.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include "../../common/utilities.h"

const MPI_Comm comm = MPI_COMM_WORLD;
const std::string file_name = "mesh";

using namespace dealii;



template<int dim>
void
test(int n_refinements, MPI_Comm comm)
{
  // serialization phase
  {
    // create pdt
    Triangulation<dim> basetria;
    GridGenerator::hyper_L(basetria);
    basetria.refine_global(n_refinements);
  
    GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm),
                                       basetria,
                                       SparsityTools::Partitioner::metis);
  
    // create instance of pft
    parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
  
    // extract relevant information form serial triangulation
    auto construction_data =
      parallel::fullydistributed::Utilities::copy_from_triangulation(basetria, tria_pft);
    
    parallel::fullydistributed::Utilities::serialize(construction_data, file_name, comm);
    
    GridOut grid_out;
    if(Utilities::MPI::this_mpi_process(comm) == 0)
      grid_out.write_mesh_per_processor_as_vtu(basetria, "tria", true, true);
  }

  // deserialization phase 
  {
    auto construction_data = parallel::fullydistributed::Utilities::deserialize<dim>(file_name, comm);
          
    // create instance of pft
    parallel::fullydistributed::Triangulation<dim> tria_pft(comm);
    // actually create triangulation
    tria_pft.reinit(construction_data);
  
    // test triangulation
    FE_Q<dim>       fe(2);
    DoFHandler<dim> dof_handler(tria_pft);
    dof_handler.distribute_dofs(fe);
  
    // output meshes as VTU
    GridOut grid_out;
    grid_out.write_mesh_per_processor_as_vtu(tria_pft, "tria_pft", true, true);
  }
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