#ifndef CACHE_H
#define CACHE_H

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <list>
#include <memory>

// Represents the set of mapped projection matrices for a single MoE expert
struct MappedExpert {
    int layer_idx;
    int expert_idx;
    
    // The safetensors file handler
    std::shared_ptr<SafetensorsFile> file_handler;
    
    // Mapped weights
    std::shared_ptr<Tensor> up_proj;
    std::shared_ptr<Tensor> down_proj;
    std::shared_ptr<Tensor> gate_proj; // Optional depending on model arch
};

// Manages the paging, memory-mapping, and un-mapping of experts using an LRU cache
class ExpertCache {
public:
    ExpertCache(const std::string& experts_dir, size_t max_experts);
    ~ExpertCache() = default;
    
    // Retrieves an expert. If it's cached, it moves to the front of LRU (hit).
    // If not, it un-maps the oldest expert (if full) and memory-maps the new one from disk (miss).
    std::shared_ptr<MappedExpert> get_expert(int layer_idx, int expert_idx);
    
    // Empties the cache and unmaps all loaded experts
    void clear();
    
    // Utility functions for debugging
    size_t get_current_size() const { return cache_map.size(); }
    void print_cache_state() const;

private:
    std::string experts_directory;
    size_t capacity;
    
    // List to maintain LRU order (front is Most Recently Used, back is Least Recently Used)
    std::list<std::shared_ptr<MappedExpert>> lru_list;
    
    // Helper to generate a unique 64-bit key from layer and expert indices
    uint64_t make_key(int layer, int expert) const {
        return (static_cast<uint64_t>(layer) << 32) | static_cast<uint32_t>(expert);
    }
    
    // Quick hash-map lookup from key to list node iterator
    std::unordered_map<uint64_t, std::list<std::shared_ptr<MappedExpert>>::iterator> cache_map;
    
    // Loads an expert from disk and maps its tensors
    std::shared_ptr<MappedExpert> load_and_map_expert(int layer, int expert);
};

#endif // CACHE_H
