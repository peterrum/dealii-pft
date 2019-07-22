

namespace dealii
{
namespace GridTools
{
void
Graph::print(std::ostream & out)
{
  std::cout << std::endl;
  std::cout << "Graph:" << std::endl;
  for(auto i : xadj)
    out << i << " ";
  out << std::endl;

  for(auto i : adjncy)
    out << i << " ";
  out << std::endl;

  for(auto i : weights)
    out << i << " ";
  out << std::endl;

  out << elements << std::endl;

  for(auto i : parts)
    out << i << " ";
  out << std::endl << std::endl;
}

void
PartitioningAlgorithm::mesh_to_dual(std::vector<int> & /*eptr_in*/,
                                    std::vector<int> & /*eind_in*/,
                                    int /*ncommon_in*/,
                                    Graph & /*graph_out*/)
{
}

void
PartitioningAlgorithm::partition(Graph & /*graph*/,
                                 int /*n_partitions*/,
                                 bool /*is_prepartitioned*/)
{
}

void
MetisPartitioningAlgorithm::mesh_to_dual(std::vector<int> & eptr_in,
                                         std::vector<int> & eind_in,
                                         int                ncommon_in,
                                         Graph &            graph_out)
{
#ifdef DEAL_II_WITH_METIS
  // extract relevant quantities
  const unsigned int n_elements = eptr_in.size() - 1;
  const unsigned int n_nodes    = *std::max_element(eind_in.begin(), eind_in.end()) + 1;

  // convert data type
  idx_t              numflag = 0;
  idx_t              ne      = n_elements;
  idx_t              nn      = n_nodes;
  idx_t              ncommon = ncommon_in;
  std::vector<idx_t> eptr    = eptr_in;
  std::vector<idx_t> eind    = eind_in;
  ;

  // perform actual conversion from mesh to dual graph
  idx_t * xadj;
  idx_t * adjncy;
  AssertThrow(METIS_MeshToDual(&ne, &nn, &eptr[0], &eind[0], &ncommon, &numflag, &xadj, &adjncy) ==
                METIS_OK,
              ExcMessage("There has been problem during METIS_MeshToDual."));

  // convert result to the right format
  auto & xadj_out = graph_out.xadj;
  xadj_out.resize(n_elements + 1);
  for(unsigned int i = 0; i < n_elements + 1; i++)
    xadj_out[i] = xadj[i];

  const unsigned int n_links    = xadj[ne];
  auto &             adjncy_out = graph_out.adjncy;
  adjncy_out.resize(n_links);
  for(unsigned int i = 0; i < n_links; i++)
    adjncy_out[i] = adjncy[i];

  graph_out.parts.resize(n_elements);
  graph_out.elements = n_elements;

  // delete temporal variables
  AssertThrow(METIS_Free(xadj) == METIS_OK,
              ExcMessage("There has been problem during METIS_Free."));
  AssertThrow(METIS_Free(adjncy) == METIS_OK,
              ExcMessage("There has been problem during METIS_Free."));
#else
  AssertThrow(false, ExcMessage("deal.II hase not been compiled with Metis."));
  (void)eptr_in;
  (void)eind_in;
  (void)ncommon_in;
  (void)graph_out;
#endif
}

void
MetisPartitioningAlgorithm::partition(Graph & graph, int n_partitions, bool is_prepartitioned)
{
#ifdef DEAL_II_WITH_METIS
  idx_t ne   = graph.elements;
  idx_t ncon = 1;
  idx_t edgecut;
  idx_t nparts = n_partitions;

  std::vector<idx_t> xadj   = graph.xadj;
  std::vector<idx_t> adjncy = graph.adjncy;
  std::vector<idx_t> parts(graph.elements);

  int status = METIS_OK;

  if(n_partitions == 1)
  {
    for(unsigned int i = 0; i < parts.size(); i++)
      parts[i] = 0;
  }
  else if(is_prepartitioned)
  {
    std::vector<idx_t> adjwgt(adjncy.size());

    // check if edge is connecting
    for(unsigned int i = 0; i < xadj.size() - 1; i++)
      for(int j = graph.xadj[i]; j < graph.xadj[i + 1]; j++)
        if(graph.parts[i] == graph.parts[graph.adjncy[j]])
          adjwgt[j] = 10;
        else
          adjwgt[j] = 1;

    status = METIS_PartGraphRecursive(&ne,
                                      &ncon,
                                      &xadj[0],
                                      &adjncy[0],
                                      NULL,
                                      NULL,
                                      &adjwgt[0],
                                      &nparts,
                                      NULL,
                                      NULL,
                                      NULL,
                                      &edgecut,
                                      &parts[0]);
  }
  else
  {
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    //          options[METIS_OPTION_MINCONN] = 1;
    //          options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_VOL;
    //          options[METIS_OPTION_NCUTS] = 10;
    //          options[METIS_OPTION_UFACTOR] = 1;
    //          options[METIS_OPTION_DBGLVL] = 1;
    status = METIS_PartGraphRecursive(&ne,
                                      &ncon,
                                      &xadj[0],
                                      &adjncy[0],
                                      NULL,
                                      NULL,
                                      NULL,
                                      &nparts,
                                      NULL,
                                      NULL,
                                      options,
                                      &edgecut,
                                      &parts[0]);
  }

  AssertThrow(status == METIS_OK, ExcMessage("Partitioning with Metis was not successful."));

  graph.parts = parts;
#else
  AssertThrow(false, ExcMessage("deal.II hase not been compiled with Metis."));
  (void)graph;
  (void)n_partitions;
#endif
}


} // namespace GridTools
} // namespace dealii