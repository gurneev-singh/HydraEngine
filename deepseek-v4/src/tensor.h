#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <cstdint>

// Supported data types in our custom engine
enum class DataType {
    F32,     // Float 32-bit (standard precision)
    F16,     // Float 16-bit (half precision)
    Q8_0,    // 8-bit quantized
    Q4_0,    // 4-bit quantized (standard block Q4_0)
    Q2_K     // 2-bit quantized (for experts)
};

// Represents a multi-dimensional tensor in memory
struct Tensor {
    std::vector<int64_t> shape;
    DataType type = DataType::F32;
    void* data = nullptr;          // Pointer to raw data buffer (RAM or GPU)
    size_t size_bytes = 0;         // Size of the raw data buffer in bytes
    bool is_mmapped = false;       // True if this memory is mapped from disk
    
    Tensor() = default;
    Tensor(const std::vector<int64_t>& s, DataType t, void* d, size_t sb, bool mmapped = false)
        : shape(s), type(t), data(d), size_bytes(sb), is_mmapped(mmapped) {}
        
    ~Tensor() {
        // Memory cleanup logic will be handled by the allocator or cache manager
    }
    
    // Helper to calculate total number of elements in the tensor
    int64_t numel() const {
        if (shape.empty()) return 0;
        int64_t count = 1;
        for (int64_t dim : shape) {
            count *= dim;
        }
        return count;
    }
};

// Safetensors header metadata for a single tensor
struct TensorMetadata {
    DataType dtype;
    std::vector<int64_t> shape;
    size_t start_offset;
    size_t end_offset;
};

// Zero-dependency C++ parser for safetensors files
class SafetensorsFile {
public:
    std::string file_path;
    std::unordered_map<std::string, TensorMetadata> tensors;
    size_t data_offset = 0; // Byte index where binary tensor data begins
    
    SafetensorsFile(const std::string& path);
    ~SafetensorsFile();
    
    // Open the file and parse the JSON header
    bool parse_header();
    
    // Memory-maps a single tensor from disk into memory
    std::shared_ptr<Tensor> map_tensor(const std::string& tensor_name);

private:
    int file_descriptor = -1;
    void* mmap_base_ptr = nullptr;
    size_t file_size = 0;
    
    // Internal helper to convert safetensors dtype string to DataType enum
    DataType parse_dtype(const std::string& dtype_str);
};

#endif // TENSOR_H
