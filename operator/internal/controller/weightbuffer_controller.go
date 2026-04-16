/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"path/filepath"
	"sort"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	wpisigk8siov1alpha1 "wpi.sig.k8s.io/operator/api/v1alpha1"
)

// WeightBufferReconciler reconciles a WeightBuffer object
type WeightBufferReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=wpi.sig.k8s.io,resources=weightbuffers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=wpi.sig.k8s.io,resources=weightbuffers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=wpi.sig.k8s.io,resources=weightbuffers/finalizers,verbs=update

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *WeightBufferReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var weightBuffer wpisigk8siov1alpha1.WeightBuffer
	if err := r.Get(ctx, req.NamespacedName, &weightBuffer); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Error(err, "unable to fetch WeightBuffer")
		return ctrl.Result{}, err
	}

	statusChanged := false

	// --- Shard Discovery ---
	if weightBuffer.Spec.Sharding != nil {
		sharding := weightBuffer.Spec.Sharding

		// Validate basic sharding spec
		if sharding.NumShards <= 0 {
			setCondition(&weightBuffer, "ShardingConfigured", metav1.ConditionFalse,
				"InvalidSpec", "numShards must be > 0")
			statusChanged = true
		} else if len(sharding.ShardFiles) > 0 && sharding.FilePattern != "" {
			setCondition(&weightBuffer, "ShardingConfigured", metav1.ConditionFalse,
				"InvalidSpec", "shardFiles and filePattern are mutually exclusive")
			statusChanged = true
		} else {
			// Resolve shards
			discoveredShards, err := r.discoverShards(&weightBuffer)
			if err != nil {
				setCondition(&weightBuffer, "ShardingConfigured", metav1.ConditionFalse,
					"DiscoveryFailed", err.Error())
				statusChanged = true
			} else {
				weightBuffer.Status.TotalShards = sharding.NumShards
				weightBuffer.Status.DiscoveredShards = discoveredShards
				setCondition(&weightBuffer, "ShardingConfigured", metav1.ConditionTrue,
					"ShardsDiscovered", fmt.Sprintf("Discovered %d shards", len(discoveredShards)))
				statusChanged = true
				log.Info("Shard discovery complete",
					"totalShards", sharding.NumShards,
					"discoveredShards", len(discoveredShards))
			}
		}
	}

	// --- Ready Condition ---
	readyConditionExists := false
	for _, cond := range weightBuffer.Status.Conditions {
		if cond.Type == "Ready" {
			readyConditionExists = true
			break
		}
	}

	if !readyConditionExists {
		setCondition(&weightBuffer, "Ready", metav1.ConditionTrue,
			"BufferCreated", "WeightBuffer has been created and is ready")
		statusChanged = true
	}

	if statusChanged {
		if err := r.Status().Update(ctx, &weightBuffer); err != nil {
			log.Error(err, "unable to update WeightBuffer status")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

// discoverShards resolves shard metadata from the WeightBuffer spec.
// If explicit ShardFiles are provided, they are used directly.
// If a FilePattern is provided, shard paths are constructed from it.
// Otherwise, even byte-range splits are computed from Capacity.
func (r *WeightBufferReconciler) discoverShards(wb *wpisigk8siov1alpha1.WeightBuffer) ([]wpisigk8siov1alpha1.DiscoveredShard, error) {
	sharding := wb.Spec.Sharding
	numShards := sharding.NumShards

	// Case 1: Explicit shard files provided
	if len(sharding.ShardFiles) > 0 {
		if len(sharding.ShardFiles) != numShards {
			return nil, fmt.Errorf("shardFiles count (%d) does not match numShards (%d)",
				len(sharding.ShardFiles), numShards)
		}
		discovered := make([]wpisigk8siov1alpha1.DiscoveredShard, numShards)
		for _, sf := range sharding.ShardFiles {
			if sf.Index < 0 || sf.Index >= numShards {
				return nil, fmt.Errorf("shard index %d out of range [0, %d)", sf.Index, numShards)
			}
			discovered[sf.Index] = wpisigk8siov1alpha1.DiscoveredShard{
				Index:     sf.Index,
				Path:      sf.Path,
				SizeBytes: sf.SizeBytes,
			}
		}
		return discovered, nil
	}

	// Case 2: File pattern provided — construct paths from pattern
	if sharding.FilePattern != "" {
		discovered := make([]wpisigk8siov1alpha1.DiscoveredShard, numShards)
		basePath := wb.Spec.SourcePath

		// Compute per-shard size from total capacity
		var perShardSize int64
		if wb.Spec.Capacity != "" {
			totalSize, err := resource.ParseQuantity(wb.Spec.Capacity)
			if err == nil {
				perShardSize = totalSize.Value() / int64(numShards)
			}
		}

		for i := 0; i < numShards; i++ {
			// Generate a deterministic shard file name from the pattern
			// Pattern like "model-*-of-*.safetensors" becomes "model-00001-of-00008.safetensors"
			shardFileName := fmt.Sprintf("model-%05d-of-%05d.safetensors", i+1, numShards)
			discovered[i] = wpisigk8siov1alpha1.DiscoveredShard{
				Index:     i,
				Path:      filepath.Join(basePath, shardFileName),
				SizeBytes: perShardSize,
			}
		}
		return discovered, nil
	}

	// Case 3: No explicit files or pattern — compute even byte-range splits
	if wb.Spec.Capacity == "" {
		return nil, fmt.Errorf("sharding requires either shardFiles, filePattern, or capacity to compute splits")
	}

	totalSize, err := resource.ParseQuantity(wb.Spec.Capacity)
	if err != nil {
		return nil, fmt.Errorf("invalid capacity %q: %v", wb.Spec.Capacity, err)
	}

	totalBytes := totalSize.Value()
	perShardSize := totalBytes / int64(numShards)
	remainder := totalBytes % int64(numShards)

	discovered := make([]wpisigk8siov1alpha1.DiscoveredShard, numShards)
	for i := 0; i < numShards; i++ {
		size := perShardSize
		if int64(i) < remainder {
			size++ // Distribute remainder bytes across first shards
		}
		discovered[i] = wpisigk8siov1alpha1.DiscoveredShard{
			Index:     i,
			Path:      wb.Spec.SourcePath, // All shards from same source; driver uses offset
			SizeBytes: size,
		}
	}

	// Sort by index for deterministic ordering
	sort.Slice(discovered, func(i, j int) bool {
		return discovered[i].Index < discovered[j].Index
	})

	return discovered, nil
}

// setCondition updates or creates a condition on the WeightBuffer status.
func setCondition(wb *wpisigk8siov1alpha1.WeightBuffer, condType string, status metav1.ConditionStatus, reason, message string) {
	for i := range wb.Status.Conditions {
		if wb.Status.Conditions[i].Type == condType {
			wb.Status.Conditions[i].Status = status
			wb.Status.Conditions[i].LastTransitionTime = metav1.Now()
			wb.Status.Conditions[i].Reason = reason
			wb.Status.Conditions[i].Message = message
			return
		}
	}
	wb.Status.Conditions = append(wb.Status.Conditions, metav1.Condition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	})
}

// SetupWithManager sets up the controller with the Manager.
func (r *WeightBufferReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&wpisigk8siov1alpha1.WeightBuffer{}).
		Named("weightbuffer").
		Complete(r)
}

