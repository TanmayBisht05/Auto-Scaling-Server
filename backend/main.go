package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"runtime"
	"time"
)

const (
	baseProcessingMS  = 30
	cpuWorkIterations = 500_000
)

// doCPUWork runs a tight loop that actually saturates a CPU core.
// With Go's goroutine scheduler, concurrent requests will spread across
// all available cores — giving the autoscaler a real CPU signal.
func doCPUWork() float64 {
	result := 0.0
	for i := 0; i < cpuWorkIterations; i++ {
		result += math.Sqrt(float64(i))
	}
	return result
}

func apiHandler(w http.ResponseWriter, r *http.Request) {
	// Simulate async I/O (non-blocking — other goroutines run during this)
	time.Sleep(time.Duration(baseProcessingMS+rand.Intn(20)) * time.Millisecond)

	result := doCPUWork()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "ok",
		"result":    result,
		"timestamp": time.Now().UnixMilli(),
	})
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
}

func main() {
	// Use all available CPU cores — this is the critical difference from Node.js
	runtime.GOMAXPROCS(runtime.NumCPU())

	port := os.Getenv("PORT")
	if port == "" {
		port = "5000"
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/api", apiHandler)
	mux.HandleFunc("/health", healthHandler)

	fmt.Printf("Go server on :%s using %d CPUs\n", port, runtime.NumCPU())
	if err := http.ListenAndServe(":"+port, mux); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}