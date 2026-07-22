#include "cache.h"
#include <iostream>
#include <sstream>

ExpertCache::ExpertCache(const std::string& experts_dir, size_t max_experts) 
    : experts_directory(experts_dir), capacity(max_experts) {}

std::shared_ptr<MappedExpert> ExpertCache::get_expert(int layer_idx, int expert_idx) {
    uint64_t key = make_key(layer_idx, expert_idx);
    auto it = cache_map.find(key);
    
    if (it != cache_map.end()) {
        // Cache Hit!
        // Move the accessed expert to the front of the LRU list (Most Recently Used)
        lru_list.splice(lru_list.begin(), lru_list, it->second);
        std::cout << "[Cache Hit] Layer " << layer_idx << " Expert " << expert_idx << " is already mapped." << std::endl;
        return *lru_list.begin();
    }
    
    // Cache Miss!
    std::cout << "[Cache Miss] Layer " << layer_idx << " Expert " << expert_idx << " needs loading..." << std::endl;
    
    // If cache is full, evict the Least Recently Used expert (at the back of the list)
    if (lru_list.size() >= capacity) {
        auto evicted_expert = lru_list.back();
        uint64_t evicted_key = make_key(evicted_expert->layer_idx, evicted_expert->expert_idx);
        
        std::cout << "[Cache Eviction] Unmapping Layer " << evicted_expert->layer_idx 
                  << " Expert " << evicted_expert->expert_idx << " to free RAM." << std::endl;
                  
        cache_map.erase(evicted_key);
        lru_list.pop_back(); // RAII unmaps the file automatically here
    }
    
    // Load and map the new expert
    auto new_expert = load_and_map_expert(layer_idx, expert_idx);
    if (!new_expert) {
        std::cerr << "[Error] Failed to load expert " << expert_idx << " for layer " << layer_idx << std::endl;
        return nullptr;
    }
    
    // Add to the front of the list and update the map
    lru_list.push_front(new_expert);
    cache_map[key] = lru_list.begin();
    
    return new_expert;
}

std::shared_ptr<MappedExpert> ExpertCache::load_and_map_expert(int layer, int expert) {
    // Construct filepath (compatible with Windows and Unix separators)
    std::stringstream ss;
    ss << experts_directory << "/expert_" << layer << "_" << expert << ".safetensors";
    std::string path = ss.str();
    
    auto file_handler = std::make_shared<SafetensorsFile>(path);
    if (!file_handler->parse_header()) {
        return nullptr;
    }
    
    auto mapped_exp = std::make_shared<MappedExpert>();
    mapped_exp->layer_idx = layer;
    mapped_exp->expert_idx = expert;
    mapped_exp->file_handler = file_handler;
    
    // Safetensors keys inside our sharded expert file:
    // e.g. "up_proj.weight", "down_proj.weight", "gate_proj.weight"
    mapped_exp->up_proj = file_handler->map_tensor("up_proj.weight");
    mapped_exp->down_proj = file_handler->map_tensor("down_proj.weight");
    
    // Check if gate projection exists (optional for some models, standard for Mistral/Qwen)
    if (file_handler->tensors.find("gate_proj.weight") != file_handler->tensors.end()) {
        mapped_exp->gate_proj = file_handler->map_tensor("gate_proj.weight");
    }
    
    return mapped_exp;
}

void ExpertCache::clear() {
    cache_map.clear();
    lru_list.clear(); // Shared pointers deleted -> destructors run -> unmapping completes
}

void ExpertCache::print_cache_state() const {
    std::cout << "--- Cache State (" << lru_list.size() << "/" << capacity << ") ---" << std::endl;
    int index = 0;
    for (const auto& exp : lru_list) {
        std::cout << "  [" << index++ << "] Layer " << exp->layer_idx 
                  << " Expert " << exp->expert_idx << std::endl;
    }
    std::cout << "---------------------------------" << std::endl;
}
