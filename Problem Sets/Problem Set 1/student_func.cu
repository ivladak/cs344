// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Grean and Blue is in it.
//The 'A' stands for Alpha and is used for transparency, it will be
//ignored in this homework.

//Each channel Red, Blue, Green and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols,
                       int block_size)
{
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int pixel_x = bx * block_size + tx;
  int pixel_y = by * block_size + ty;

  if (pixel_x >= numCols || pixel_y >= numRows)
    return;
  int pixel_idx = pixel_y * numCols + pixel_x;

  uchar4 rgba = rgbaImage[pixel_idx];
  greyImage[pixel_idx] = .299f * rgba.x + .587f * rgba.y + .114f * rgba.z;
}

int round_next_pow2(int x, int min_pow)
{
  int l, m, r;
  l = min_pow - 1;
  r = 31;
  while (r - l > 1) {
    m = (r + l) / 2;
    if ((1 << m) < x)
      l = m;
    else
      r = m;
  }
  return 1 << r;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const int block_size = 32;
  const int block_size_log2 = 5;
  const dim3 blockSize(block_size, block_size); // 1024 threads per block
  int numRowsRounded = round_next_pow2(numRows, block_size_log2);
  int numColsRounded = round_next_pow2(numCols, block_size_log2);
  const dim3 gridSize(numColsRounded / block_size, numRowsRounded / block_size);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols, block_size);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

}
