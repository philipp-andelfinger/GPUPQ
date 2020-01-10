#pragma once


// replacement for std::bitset for cuda
template<size_t N> class cuda_bitset {

public:
  // MAYBE: implement index operator overloading instead of set / get, that would need a reference type though.
  // See http://en.cppreference.com/w/cpp/utility/bitset for the interface.

  __host__ __device__ bool get(size_t index) {
    return bitset[index];
  }

  __host__ __device__ void set(size_t index, bool value) {
    bitset[index] = value;
  }

  __host__ __device__ void clear() {
    memset(&bitset, 0, sizeof(bitset));
  }


private:
  bool bitset[N];
};
