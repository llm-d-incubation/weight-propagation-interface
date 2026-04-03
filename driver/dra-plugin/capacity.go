package main

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// getTotalVRAM queries nvidia-smi for total GPU memory in bytes.
// If it fails, it returns an error.
func getTotalVRAM() (uint64, error) {
	cmd := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader")
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return 0, fmt.Errorf("nvidia-smi error: %v", err)
	}

	// Output format: "81920 MiB\n", we might have multiple GPUs, grab the first one
	output := strings.TrimSpace(out.String())
	lines := strings.Split(output, "\n")
	if len(lines) == 0 || lines[0] == "" {
		return 0, fmt.Errorf("empty output from nvidia-smi")
	}

	parts := strings.Fields(lines[0])
	if len(parts) == 0 {
		return 0, fmt.Errorf("unexpected nvidia-smi output: %s", lines[0])
	}

	val, err := strconv.ParseUint(parts[0], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse memory value '%s': %v", parts[0], err)
	}

	// Standard nvidia-smi output in MiB
	return val * 1024 * 1024, nil
}

// publishResourceSliceLoop continuously publishes the node's ResourceSlice.
func publishResourceSliceLoop(ctx context.Context, dynClient dynamic.Interface, nodeName string) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	rsGVR := schema.GroupVersionResource{Group: "resource.k8s.io", Version: "v1", Resource: "resourceslices"}
	rsName := "wpi-node-" + nodeName

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			vramBytes, err := getTotalVRAM()
			if err != nil {
				log.Printf("Failed to get VRAM bounds, skipping ResourceSlice update: %v", err)
				continue
			}

			// Construct unstructured ResourceSlice
			slice := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "resource.k8s.io/v1",
					"kind":       "ResourceSlice",
					"metadata": map[string]interface{}{
						"name": rsName,
					},
					"spec": map[string]interface{}{
						"driver":   "wpi.sig.k8s.io",
						"nodeName": nodeName,
						"pool": map[string]interface{}{
							"name":               nodeName,
							"generation":         int64(1),
							"resourceSliceCount": int64(1),
						},
						"devices": []interface{}{
							map[string]interface{}{
								"name": "wpi-device-0",
								"capacity": map[string]interface{}{
									"memory": map[string]interface{}{
										"value": fmt.Sprintf("%d", vramBytes),
									},
								},
							},
						},
					},
				},
			}

			// Upsert logical
			existing, err := dynClient.Resource(rsGVR).Get(ctx, rsName, metav1.GetOptions{})
			if err == nil {
				// Update
				slice.SetResourceVersion(existing.GetResourceVersion())
				_, updateErr := dynClient.Resource(rsGVR).Update(ctx, slice, metav1.UpdateOptions{})
				if updateErr != nil {
					log.Printf("Failed to update ResourceSlice %s: %v", rsName, updateErr)
				}
			} else {
				// Create
				_, createErr := dynClient.Resource(rsGVR).Create(ctx, slice, metav1.CreateOptions{})
				if createErr != nil {
					log.Printf("Failed to create ResourceSlice %s: %v", rsName, createErr)
				}
			}
		}
	}
}
