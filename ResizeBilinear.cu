/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "ResizeBilinear.hpp"
#include <cuda_fp16.h>
#include <cassert>

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims ResizeBilinearPlugin::getOutputDimensions(int index,
                                                        const nvinfer1::Dims *inputDims,
                                                        int nbInputs) {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(is_CHW(input));
  assert(_ndims == 2);
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  int s = 0;
  for( int d=0; d<input.nbDims; ++d ) {
    output.type[d] = input.type[d];
    if( input.type[d] == nvinfer1::DimensionType::kSPATIAL ) {
      output.d[d] = int(input.d[d] * _scale[s++]);
    } else {
      output.d[d] = input.d[d];
    }
  }
  return output;
}

int ResizeBilinearPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  assert(is_CHW(this->getInputDims(0)));
  assert(is_CHW(_output_dims));
  assert(_ndims == 2);
  return 0;
}

template <typename Data>
__global__
void resize_bilinear_kernel_2d(int nbatch,
                              float2 rate,
                              Data const* idata, int iwidth, int iheight,  int ibatchstride,
                              Data*       odata, int owidth, int oheight, int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;
  for( int batch=z0; batch<nbatch; batch+=gridDim.z ) {
    for( int oy=y0; oy<oheight; oy+=blockDim.y*gridDim.y ) {
      for( int ox=x0; ox<owidth; ox+=blockDim.x*gridDim.x ) {
        const float iyr = rate.y * oy;
        const int iy = (int)iyr;
        const int iyp = (iy < iheight - 1) ? 1 : 0;
        const float h1lambda = iyr - iy;
        const float h0lambda = 1. - h1lambda;
        
        const float ixr = rate.x * ox;
        const int ix = (int)ixr;
        const int ixp = (ix < iwidth - 1) ? 1 : 0;
        const float w1lambda = ixr - ix;
        const float w0lambda = 1. - w1lambda;

        const float nb0 = idata[batch * ibatchstride + iy * iwidth + ix]; 
        const float nb1 = idata[batch * ibatchstride + iy * iwidth + (ix + ixp)];
        const float nb2 = idata[batch * ibatchstride + (iy + iyp) * iwidth + ix]; 
        const float nb3 = idata[batch * ibatchstride + (iy + iyp) * iwidth + (ix + ixp)]; 

        odata[batch * obatchstride + oy * owidth + ox] = 
            h0lambda * (w0lambda * nb0 + w1lambda * nb1) +
            h1lambda * (w0lambda * nb2 + w1lambda * nb3);
        
        //odata[batch * obatchstride + oy * owidth + ox] =
        //    h0lambda * (
        //        w0lambda *
        //            idata[batch * ibatchstride + iy * iwidth + ix] 
        //        +
        //        w1lambda *
        //            idata[batch * ibatchstride + iy * iwidth + (ix + ixp)]
        //    ) 
        //    +
        //    h1lambda * (
        //        w0lambda * 
        //            idata[batch * ibatchstride + (iy + iyp) * iwidth + ix] 
        //        + 
        //        w1lambda * 
        //            idata[batch * ibatchstride + (iy + iyp) * iwidth + (ix + ixp)] 
        //            
        //    );
      }
    }
  }
}

int ResizeBilinearPlugin::enqueue(int batchSize,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int nchan = input_dims.d[0];
  switch( _ndims ) {
  case 2: {

    int iwidth =   input_dims.d[2];
    int owidth = _output_dims.d[2];
    int iheight = input_dims.d[1];
    int oheight = _output_dims.d[1];


    float xrate = owidth > 1 ? (iwidth - 1.0) / (owidth - 1.0) : 0.f;
    float yrate = oheight > 1 ? (iheight - 1.0) / (oheight - 1.0) : 0.f;

    float2 rate = {xrate, yrate};

    int ibatchstride =   iheight * iwidth;
    int obatchstride = oheight * owidth;

    dim3 block(32, 16);
    dim3 grid((owidth - 1) / block.x + 1,
              (oheight - 1) / block.y + 1,
              std::min(batchSize * nchan, 65535));
    if (getDataType()==nvinfer1::DataType::kFLOAT) {				
      resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, rate,
         static_cast<float const*>( inputs[0]), iwidth, iheight, ibatchstride,
         static_cast<float*      >(outputs[0]), owidth, oheight, obatchstride);
    } else {
      resize_bilinear_kernel_2d<<<grid, block, 0, stream>>>
        (batchSize * nchan, rate,
         static_cast<__half const*>( inputs[0]), iwidth, iheight, ibatchstride,
         static_cast<__half*      >(outputs[0]), owidth, oheight, obatchstride);
    }
    return cudaGetLastError() != cudaSuccess;
  }
  default: return -1;
  }
}
