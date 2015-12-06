#pragma once

#include "CUDADefinitions.h"
#include "CUDAStreams.h"
#include "Timer.h"

/**
* @class TimerCUDA
* CUDA implementation of the Timer interface
*/
class TimerCUDA {
public:    
    Timer() {
        Reset();
        // create the CUDA events
        cudaEventCreate(&start_); 
        cudaEventCreate(&stop_);
    }
    
    ~Timer() {
        // free the CUDA events 
        cudaEventDestroy(start_); 
        cudaEventDestroy(stop_); 
    }

    /**
    * Reset counters
    */
    void Reset() {
      totalTime_ = 0.0;
    }

    /**
    * Start the stop watch
    */
    void Start()
    {
        // insert a start event
        cudaEventRecord(start_, 0);
    }

    /**
    * Pause the stop watch
    */
    double Pause()
    {
        // insert stop event and wait for it
        cudaEventRecord(stop_, 0); 
        cudaEventSynchronize(stop_);
        
        // compute the timing
        float result;
        cudaEventElapsedTime(&result, start_, stop_);
        return static_cast<double>(result) * 0.001f; // convert ms to s
    }
    
    double totalTime() const
    {
        return totalTime_;
    }
    
    std::string ToString() const
    {
        std::ostringstream out; 
        out << name_ << "\t[s]\t" << totalTime_; 
        return out.str();
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
    
    std::string name_ = "Timer";
    double totalTime_;
};

  

