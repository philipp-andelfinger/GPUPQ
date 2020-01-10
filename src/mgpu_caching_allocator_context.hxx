#pragma once

#include <map>

#include <moderngpu/context.hxx>

// caching allocator for device memory
class caching_allocator_context_t : public mgpu::standard_context_t {

public:
  caching_allocator_context_t(cudaStream_t stream_ = 0) :
    standard_context_t(false, stream_) {
    num_hits = num_allocations = 0;
    num_alloc_blocks = 0;
  }


  ~caching_allocator_context_t() {
    printf("allocator: %.1f%% cache hits\n", 100.0 * num_hits / num_allocations);

    // deallocate all outstanding blocks in both lists
    for(auto i = unused_allocations.begin(); i != unused_allocations.end(); ++i)
    {
      mgpu::standard_context_t::free(i->second, mgpu::memory_space_device);
    }

    for(auto i = used_allocations.begin(); i != used_allocations.end(); ++i)
    {
      mgpu::standard_context_t::free(i->first, mgpu::memory_space_device);
    }
  }

  virtual void* alloc(size_t size, mgpu::memory_space_t space) {
    num_allocations++;

    void* ptr = nullptr;

    assert(space == mgpu::memory_space_device);

    if (size != 0) {

      // ceil to next integer to get number of required blocks
      int blocks = (size + mem_block_size - 1) / mem_block_size;
      // printf("Requested %d bytes, ceiled to %d (%d blocks). (%d/%d blocks used)... ",
      //    size, blocks * mem_block_size, blocks, num_alloc_blocks, max_alloc_blocks);


      // search the cache for a free allocation of the exactly right size
      auto allocation = unused_allocations.find(blocks);

      if(allocation != unused_allocations.end())
      {
        // there is an unused allocation of the right size, use it.
        num_hits++;
        // printf("Cache hit!\n");

        ptr = allocation->second;

        unused_allocations.erase(allocation);
      }
      else
      {
        // no allocation of the right size exists, allocate a new one
        ptr = mgpu::standard_context_t::alloc(blocks * mem_block_size, space);
        // printf("Cache miss.\n");

        num_alloc_blocks += blocks;
        // throw all unused allocations away if we surpassed the maximum limit.
        // afterwards we still might use to much blocks, but this is ignored here.
        if(num_alloc_blocks > max_alloc_blocks)
        {
          release_unused();
        }
      }

      used_allocations.insert(std::make_pair(ptr, blocks));
    }

    return ptr;
  }

  // Releases the given allocation, but stores it as unused to be reused with later allocations
  virtual void free(void* ptr, mgpu::memory_space_t space) {
    if(ptr) {
      assert(space == mgpu::memory_space_device);

      // erase the allocated block from the allocated blocks map
      auto allocation = used_allocations.find(ptr);
      int blocks = allocation->second;
      used_allocations.erase(allocation);

      // printf("Freed %d blocks. (%d/%d blocks used)\n", blocks, num_alloc_blocks, max_alloc_blocks);

      // insert the block into the free blocks map
      unused_allocations.insert(std::make_pair(blocks, ptr));
    }
  }

private:
  // only for cache statistics, not necessary for the allocator's operations
  int num_hits, num_allocations;

  // allocations in number of blocks
  typedef std::multimap<int, void*> unused_allocations_type;
  typedef std::map<void*, int>     used_allocations_type;

  unused_allocations_type unused_allocations;
  used_allocations_type   used_allocations;


  // count everything in blocks, as the actual size of one block is irrelevant for the allocation computations.
  const size_t mem_block_size = HQ_CACHING_ALLOCATOR_MEM_BLOCK_SIZE;
  long num_alloc_blocks;
  const long max_alloc_blocks = HQ_CACHING_ALLOCATOR_MAX_ALLOC_BLOCKS;


  // release all unused cached blocks
  void release_unused() {
    // printf("%d/%d blocks used, cleaning up... ", num_alloc_blocks, max_alloc_blocks);
    for(auto i = unused_allocations.begin(); i != unused_allocations.end(); ++i)
    {
      num_alloc_blocks -= i->first;
      mgpu::standard_context_t::free(i->second, mgpu::memory_space_device);
    }
    unused_allocations.clear();
    // printf("%d/%d blocks used.\n", num_alloc_blocks, max_alloc_blocks);
  }

};
