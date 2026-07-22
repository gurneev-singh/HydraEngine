#include "tensor.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

SafetensorsFile::SafetensorsFile(const std::string& path) : file_path(path) {}

SafetensorsFile::~SafetensorsFile() {
    // Clean up mapping and file handles
#ifdef _WIN32
    if (mmap_base_ptr) {
        UnmapViewOfFile(mmap_base_ptr);
    }
    if (file_descriptor != -1) {
        CloseHandle(reinterpret_cast<HANDLE>(file_descriptor));
    }
#else
    if (mmap_base_ptr && mmap_base_ptr != MAP_FAILED) {
        munmap(mmap_base_ptr, file_size);
    }
    if (file_descriptor != -1) {
        close(file_descriptor);
    }
#endif
}

DataType SafetensorsFile::parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "F32") return DataType::F32;
    if (dtype_str == "F16") return DataType::F16;
    if (dtype_str == "I8" || dtype_str == "Q8_0") return DataType::Q8_0;
    if (dtype_str == "Q4_0") return DataType::Q4_0;
    if (dtype_str == "Q2_K") return DataType::Q2_K;
    return DataType::F32; // Default fallback
}

// Zero-dependency header parser using simple string tokenization
bool SafetensorsFile::parse_header() {
#ifdef _WIN32
    HANDLE hFile = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "[Error] Failed to open file: " << file_path << std::endl;
        return false;
    }
    file_descriptor = reinterpret_cast<intptr_t>(hFile);
    
    LARGE_INTEGER fs;
    if (!GetFileSizeEx(hFile, &fs)) {
        std::cerr << "[Error] Failed to get size of: " << file_path << std::endl;
        return false;
    }
    file_size = static_cast<size_t>(fs.QuadPart);
#else
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "[Error] Failed to open file: " << file_path << std::endl;
        return false;
    }
    file_descriptor = fd;
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "[Error] Failed to get size of: " << file_path << std::endl;
        return false;
    }
    file_size = sb.st_size;
#endif

    if (file_size < 8) return false;

    // Memory map the entire file for direct pointer access
#ifdef _WIN32
    HANDLE hMapping = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        std::cerr << "[Error] CreateFileMapping failed for: " << file_path << std::endl;
        return false;
    }
    mmap_base_ptr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping); // Can close mapping handle after MapView
    if (!mmap_base_ptr) {
        std::cerr << "[Error] MapViewOfFile failed for: " << file_path << std::endl;
        return false;
    }
#else
    mmap_base_ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mmap_base_ptr == MAP_FAILED) {
        std::cerr << "[Error] mmap failed for: " << file_path << std::endl;
        mmap_base_ptr = nullptr;
        return false;
    }
#endif

    // Read the first 8 bytes as little-endian uint64
    uint64_t header_len = 0;
    std::memcpy(&header_len, mmap_base_ptr, 8);
    data_offset = 8 + header_len;

    if (data_offset > file_size) {
        std::cerr << "[Error] Invalid header length in: " << file_path << std::endl;
        return false;
    }

    // Convert header section to a string for parsing
    std::string header_str(static_cast<char*>(mmap_base_ptr) + 8, header_len);

    // Simple JSON tokenizer loop
    size_t pos = 0;
    while (true) {
        // Find next tensor key (enclosed in double quotes)
        size_t key_start = header_str.find('"', pos);
        if (key_start == std::string::npos) break;
        key_start += 1;
        
        size_t key_end = header_str.find('"', key_start);
        if (key_end == std::string::npos) break;
        
        std::string tensor_name = header_str.substr(key_start, key_end - key_start);
        pos = key_end + 1;
        
        // Skip metadata tags
        if (tensor_name == "__metadata__") {
            continue;
        }

        // Find the dictionary body for this tensor: e.g., {"dtype":"F32","shape":[1024],"data_offsets":[0,4096]}
        size_t body_start = header_str.find('{', pos);
        size_t body_end = header_str.find('}', body_start);
        if (body_start == std::string::npos || body_end == std::string::npos) break;
        
        std::string body = header_str.substr(body_start, body_end - body_start + 1);
        pos = body_end + 1;

        // Parse dtype: e.g., "dtype":"F32"
        size_t dtype_pos = body.find("\"dtype\"");
        size_t dtype_val_start = body.find('"', dtype_pos + 7) + 1;
        size_t dtype_val_end = body.find('"', dtype_val_start);
        std::string dtype_str = body.substr(dtype_val_start, dtype_val_end - dtype_val_start);

        // Parse shape: e.g., "shape":[1024,4096]
        size_t shape_pos = body.find("\"shape\"");
        size_t shape_val_start = body.find('[', shape_pos) + 1;
        size_t shape_val_end = body.find(']', shape_val_start);
        std::string shape_str = body.substr(shape_val_start, shape_val_end - shape_val_start);
        
        std::vector<int64_t> shape;
        if (!shape_str.empty()) {
            std::stringstream ss(shape_str);
            std::string dim;
            while (std::getline(ss, dim, ',')) {
                shape.push_back(std::stoll(dim));
            }
        } else {
            shape.push_back(1); // Scalar fallback
        }

        // Parse offsets: e.g., "data_offsets":[0,4096]
        size_t offset_pos = body.find("\"data_offsets\"");
        size_t offset_val_start = body.find('[', offset_pos) + 1;
        size_t offset_val_end = body.find(']', offset_val_start);
        std::string offset_str = body.substr(offset_val_start, offset_val_end - offset_val_start);
        
        size_t comma = offset_str.find(',');
        size_t start_off = std::stoull(offset_str.substr(0, comma));
        size_t end_off = std::stoull(offset_str.substr(comma + 1));

        // Add to mapped tensors
        TensorMetadata meta = {
            parse_dtype(dtype_str),
            shape,
            start_off,
            end_off
        };
        tensors[tensor_name] = meta;
    }
    
    return true;
}

// Map tensor memory to a shared pointer
std::shared_ptr<Tensor> SafetensorsFile::map_tensor(const std::string& tensor_name) {
    auto it = tensors.find(tensor_name);
    if (it == tensors.end()) {
        std::cerr << "[Error] Tensor not found in file: " << tensor_name << std::endl;
        return nullptr;
    }
    
    const auto& meta = it->second;
    size_t tensor_start = data_offset + meta.start_offset;
    size_t tensor_size = meta.end_offset - meta.start_offset;
    
    if (tensor_start + tensor_size > file_size) {
        std::cerr << "[Error] Tensor offsets out of file boundaries: " << tensor_name << std::endl;
        return nullptr;
    }
    
    // Calculate raw pointer address directly on the memory-mapped file
    void* tensor_data_ptr = static_cast<char*>(mmap_base_ptr) + tensor_start;
    
    return std::make_shared<Tensor>(
        meta.shape,
        meta.dtype,
        tensor_data_ptr,
        tensor_size,
        true // marked as mmapped memory
    );
}
