// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Wrapper to adapt shmspsc.h lock-free queue to MpiMessageQueue interface

#pragma once

#include "tensorrt_llm/common/shmspsc.h"
#include "tensorrt_llm/executor/orchestratorUtils.h"
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>

namespace tensorrt_llm::executor
{

// Adapter class to make shmspsc.h compatible with MpiMessageQueue interface
class ShmMessageQueue
{
public:
    ShmMessageQueue()
    {
        // Create unique SHM name based on process ID
        std::string queueName = "/trtllm_mpi_queue_" + std::to_string(getpid());

        // Try to create as producer first
        try
        {
            queue_ = std::make_unique<QueueType>(QueueType::create(queueName.c_str()));
            isProducer_ = true;
        }
        catch (...)
        {
            // If creation fails, try attach as consumer
            try
            {
                queue_ = std::make_unique<QueueType>(QueueType::attach(queueName.c_str()));
                isProducer_ = false;
            }
            catch (std::exception const& e)
            {
                throw std::runtime_error(std::string("Failed to create/attach SHM queue: ") + e.what());
            }
        }
    }

    ~ShmMessageQueue()
    {
        // Queue cleanup handled by ShmSpscQueue destructor
    }

    // Match MpiMessageQueue::push interface
    void push(MpiMessage&& message)
    {
        queue_->publish([&](MpiMessage* slot) {
            new (slot) MpiMessage(std::move(message));
        });
    }

    // Match MpiMessageQueue::pop interface (blocking)
    MpiMessage pop()
    {
        std::optional<MpiMessage> result;

        while (!result.has_value())
        {
            queue_->process([&](MpiMessage const& msg, int64_t seq) {
                if (!result.has_value()) {
                    result = msg;  // Copy construct into optional
                }
            });

            if (!result.has_value())
            {
                // Busy spin with CPU pause instruction
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#else
                // Fallback for other architectures
                std::this_thread::yield();
#endif
            }
        }

        return std::move(*result);
    }

private:
    using QueueType = trtllm_poc::ShmSpscQueue<MpiMessage, 4096>;
    std::unique_ptr<QueueType> queue_;
    bool isProducer_ = false;
};

} // namespace tensorrt_llm::executor
