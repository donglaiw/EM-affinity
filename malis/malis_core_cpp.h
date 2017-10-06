#ifndef MALIS_CPP_H
#define MALIS_CPP_H

void preCompute(const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
        uint64_t* pre_ve, uint64_t* pre_prodDims, int32_t* pre_nHood);

void malis_loss_weights_cpp_pair(const uint64_t* seg,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, float* nPairPerEdge,
               const uint64_t* pre_ve, const uint64_t* pre_prodDims, const int32_t* pre_nHood);

void malis_loss_weights_cpp_pos(const uint64_t* seg,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, float* nPairPerEdge,
               const uint64_t* pre_ve, const uint64_t* pre_prodDims, const int32_t* pre_nHood, const int pos);

void malis_loss_weights_cpp(const uint64_t* seg,
               const uint64_t* conn_dims, const int32_t* nhood_data, const uint64_t* nhood_dims,
               const float* edgeWeight, const int pos,
               float* nPairPerEdge);

// utility function
void connected_components_cpp(const int nVert,
               const int nEdge, const uint64_t* node1, const uint64_t* node2, const int* edgeWeight,
               uint64_t* seg);

void marker_watershed_cpp(const int nVert, const uint64_t* marker,
               const int nEdge, const uint64_t* node1, const uint64_t* node2, const float* edgeWeight,
               uint64_t* seg);

#endif
