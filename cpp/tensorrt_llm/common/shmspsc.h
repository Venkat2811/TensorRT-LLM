// shmspsc.h - Minimal SHM SPSC Ring Buffer for TensorRT-LLM PoC
// Based on LMAX Disruptor pattern with cache-line padding
#pragma once

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>
#include <new>
#include <type_traits>

namespace trtllm_poc {

// Cache-line aligned to prevent false sharing
struct alignas(64) Cursor {
    std::atomic<int64_t> sequence{-1};
};

// SHM layout: [producer_cursor | consumer_cursor | buffer...]
template<typename T, size_t Capacity>
struct ShmLayout {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");
    Cursor producer;
    Cursor consumer;
    alignas(64) typename std::aligned_storage<sizeof(T), alignof(T)>::type buffer[Capacity];

    // Get typed pointer to buffer slot
    T* slot(size_t index) {
        return reinterpret_cast<T*>(&buffer[index]);
    }
    const T* slot(size_t index) const {
        return reinterpret_cast<const T*>(&buffer[index]);
    }
};

template<typename T, size_t Capacity = 1024>
class ShmSpscQueue {
    static constexpr size_t kMask = Capacity - 1;
    using Layout = ShmLayout<T, Capacity>;

    Layout* layout_ = nullptr;
    int fd_ = -1;
    bool owner_ = false;

public:
    // Producer creates
    static ShmSpscQueue create(const char* name) {
        ShmSpscQueue q;
        q.owner_ = true;

        // Remove stale
        shm_unlink(name);

        q.fd_ = shm_open(name, O_CREAT | O_RDWR, 0666);
        if (q.fd_ < 0) throw std::runtime_error("shm_open failed");

        if (ftruncate(q.fd_, sizeof(Layout)) < 0) {
            close(q.fd_);
            throw std::runtime_error("ftruncate failed");
        }

        void* ptr = mmap(nullptr, sizeof(Layout), PROT_READ | PROT_WRITE,
                         MAP_SHARED, q.fd_, 0);
        if (ptr == MAP_FAILED) {
            close(q.fd_);
            throw std::runtime_error("mmap failed");
        }

        q.layout_ = new (ptr) Layout{};  // Placement new, zero-init atomics
        return q;
    }

    // Consumer attaches
    static ShmSpscQueue attach(const char* name) {
        ShmSpscQueue q;
        q.owner_ = false;

        q.fd_ = shm_open(name, O_RDWR, 0666);
        if (q.fd_ < 0) throw std::runtime_error("shm_open failed");

        void* ptr = mmap(nullptr, sizeof(Layout), PROT_READ | PROT_WRITE,
                         MAP_SHARED, q.fd_, 0);
        if (ptr == MAP_FAILED) {
            close(q.fd_);
            throw std::runtime_error("mmap failed");
        }

        q.layout_ = static_cast<Layout*>(ptr);
        return q;
    }

    ~ShmSpscQueue() {
        if (layout_) munmap(layout_, sizeof(Layout));
        if (fd_ >= 0) close(fd_);
        // Owner doesn't unlink - let it persist for consumer
    }

    // === Producer API ===

    // Publish with functor (zero-copy write)
    template<typename F>
    void publish(F&& writer) {
        int64_t seq = layout_->producer.sequence.load(std::memory_order_relaxed) + 1;

        // Wait for slot (busy spin)
        while (seq - Capacity > layout_->consumer.sequence.load(std::memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();  // x86 spin hint
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");  // ARM yield
#endif
        }

        // Write directly to slot - writer does placement construction
        T* slot_ptr = layout_->slot(seq & kMask);
        writer(slot_ptr);  // Writer constructs object at this location

        // Publish
        layout_->producer.sequence.store(seq, std::memory_order_release);
    }

    // === Consumer API ===

    // Process available events (returns count processed)
    template<typename F>
    int64_t process(F&& handler) {
        int64_t consumer_seq = layout_->consumer.sequence.load(std::memory_order_relaxed);
        int64_t available = layout_->producer.sequence.load(std::memory_order_acquire);

        int64_t count = 0;
        while (consumer_seq < available) {
            consumer_seq++;
            const T* slot_ptr = layout_->slot(consumer_seq & kMask);
            handler(*slot_ptr, consumer_seq);
            slot_ptr->~T();  // Destroy after processing
            count++;
        }

        if (count > 0) {
            layout_->consumer.sequence.store(consumer_seq, std::memory_order_release);
        }
        return count;
    }

    // Blocking consume (busy spin)
    template<typename F>
    void consume_blocking(F&& handler) {
        while (true) {
            if (process(handler) == 0) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }
        }
    }
};

} // namespace trtllm_poc
