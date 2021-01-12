/***************************************************************************
 *
 *   Copyright (C) 2012 by Ben Barsdell and Andrew Jameson
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#pragma once

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>

#include <map>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <sstream>

#include <fmt/format.h>

/*
    This was copied from Thrust's custom_temporary_allocation example.
    https://github.com/NVIDIA/thrust/blob/master/examples/cuda/custom_temporary_allocation.cu
*/

// A simple allocator for caching cudaMalloc allocations.
struct cached_allocator {
    typedef char value_type;

    cached_allocator() {}

    ~cached_allocator() { free_all(); }

    char* allocate(std::ptrdiff_t num_bytes) {
        // fmt::print("cached_allocator::allocate(): num_bytes == {}\n",
        //           num_bytes);

        char* result = 0;

        // search the cache for a free block
        free_blocks_type::iterator free_block = free_blocks.find(num_bytes);

        if (free_block != free_blocks.end()) {
            // fmt::print("cached_allocator::allocate(): found a free block\n");

            result = free_block->second;

            // Erase from the `free_blocks` map.
            free_blocks.erase(free_block);
        } else {
            // No allocation of the right size exists, so create a new one with
            // `thrust::cuda::malloc`.
            try {
                // fmt::print(
                //    "cached_allocator::allocate(): allocating new block\n");

                // Allocate memory and convert the resulting
                // `thrust::cuda::pointer` to a raw pointer.
                result = thrust::cuda::malloc<char>(num_bytes).get();
            } catch (std::runtime_error&) {
                throw;
            }
        }

        // Insert the allocated pointer into the `allocated_blocks` map.
        allocated_blocks.insert(std::make_pair(result, num_bytes));

        return result;
    }

    void deallocate(char* ptr, size_t n) {
        // fmt::print("cached_allocator::deallocate(): ptr == {}\n",
        //           fmt::ptr(ptr));

        // Erase the allocated block from the allocated blocks map.
        allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
        if (iter == allocated_blocks.end()) {
            std::string msg = fmt::format(
                "Pointer `{}` was not allocated by this allocator.",
                fmt::ptr(ptr));
            throw std::range_error(msg);
        }

        std::ptrdiff_t num_bytes = iter->second;
        allocated_blocks.erase(iter);

        // Insert the block into the free blocks map.
        free_blocks.insert(std::make_pair(num_bytes, ptr));
    }

private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;

    free_blocks_type free_blocks;
    allocated_blocks_type allocated_blocks;

    void free_all() {
        // fmt::print("cached_allocator::free_all()\n");

        // Deallocate all outstanding blocks in both lists.
        for (free_blocks_type::iterator i = free_blocks.begin();
             i != free_blocks.end(); ++i) {
            // Transform the pointer to cuda::pointer before calling cuda::free.
            thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
        }

        for (allocated_blocks_type::iterator i = allocated_blocks.begin();
             i != allocated_blocks.end(); ++i) {
            // Transform the pointer to cuda::pointer before calling cuda::free.
            thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
        }
    }
};
