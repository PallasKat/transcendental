__host__ __device__ int fooMax(int a, int b);

__host__ __device__ int fooMax(int a, int b) {
  return ( a > b ) ? a : b;
}

