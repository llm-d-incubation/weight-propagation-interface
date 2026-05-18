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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	resourcev1 "k8s.io/api/resource/v1"
	wpisigk8siov1alpha1 "wpi.io/operator/api/v1alpha1"
)

// WeightClaimReconciler reconciles a WeightClaim object
type WeightClaimReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=wpi.io,resources=weightclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=wpi.io,resources=weightclaims/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=wpi.io,resources=weightclaims/finalizers,verbs=update
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *WeightClaimReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var weightClaim wpisigk8siov1alpha1.WeightClaim
	if err := r.Get(ctx, req.NamespacedName, &weightClaim); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Error(err, "unable to fetch WeightClaim")
		return ctrl.Result{}, err
	}

	var weightBuffer wpisigk8siov1alpha1.WeightBuffer
	bufferName := client.ObjectKey{
		Namespace: req.Namespace,
		Name:      weightClaim.Spec.WeightBufferName,
	}

	validatedStatus := metav1.ConditionTrue
	reason := "BufferFound"
	message := "Referenced WeightBuffer was found"

	if err := r.Get(ctx, bufferName, &weightBuffer); err != nil {
		if apierrors.IsNotFound(err) {
			validatedStatus = metav1.ConditionFalse
			reason = "BufferNotFound"
			message = "Referenced WeightBuffer was not found"
		} else {
			log.Error(err, "unable to fetch WeightBuffer")
			return ctrl.Result{}, err
		}
	}

	// --- Shard Validation & Assignment ---
	if validatedStatus == metav1.ConditionTrue && weightBuffer.Spec.Sharding != nil {
		shardResult := r.resolveShardIndex(&weightClaim, &weightBuffer)
		if shardResult.err != nil {
			validatedStatus = metav1.ConditionFalse
			reason = "ShardValidationFailed"
			message = shardResult.err.Error()
		} else {
			weightClaim.Status.AssignedShardIndex = &shardResult.index
			message = fmt.Sprintf("Validated. Assigned shard %d of %d (path: %s, size: %d bytes)",
				shardResult.index, weightBuffer.Status.TotalShards,
				shardResult.path, shardResult.sizeBytes)
			log.Info("Shard assigned to claim",
				"claim", weightClaim.Name,
				"shardIndex", shardResult.index,
				"shardPath", shardResult.path,
				"shardSize", shardResult.sizeBytes)
		}
	}

	// Check if Validated condition already exists and has the correct status
	var existingCond *metav1.Condition
	for i := range weightClaim.Status.Conditions {
		if weightClaim.Status.Conditions[i].Type == "Validated" {
			existingCond = &weightClaim.Status.Conditions[i]
			break
		}
	}

	if existingCond == nil || existingCond.Status != validatedStatus {
		if existingCond == nil {
			weightClaim.Status.Conditions = append(weightClaim.Status.Conditions, metav1.Condition{
				Type: "Validated",
			})
			existingCond = &weightClaim.Status.Conditions[len(weightClaim.Status.Conditions)-1]
		}

		existingCond.Status = validatedStatus
		existingCond.LastTransitionTime = metav1.Now()
		existingCond.Reason = reason
		existingCond.Message = message

		if err := r.Status().Update(ctx, &weightClaim); err != nil {
			log.Error(err, "unable to update WeightClaim status")
			return ctrl.Result{}, err
		}
	}

	// Create ResourceClaim if it doesn't exist and the buffer was found
	if validatedStatus == metav1.ConditionTrue {
		resourceClaim := &resourcev1.ResourceClaim{}
		err := r.Get(ctx, req.NamespacedName, resourceClaim)
		if err != nil && apierrors.IsNotFound(err) {
			// Define the ResourceClaim
			resourceClaim = &resourcev1.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{
					Name:      weightClaim.Name,
					Namespace: weightClaim.Namespace,
				},
				Spec: resourcev1.ResourceClaimSpec{
					Devices: resourcev1.DeviceClaim{
						Requests: []resourcev1.DeviceRequest{
							{
								Name: "wpi-device",
								Exactly: &resourcev1.ExactDeviceRequest{
									DeviceClassName: "wpi-device-class",
								},
							},
						},
					},
				},
			}
			// Set owner reference
			if err := ctrl.SetControllerReference(&weightClaim, resourceClaim, r.Scheme); err != nil {
				log.Error(err, "unable to set owner reference on ResourceClaim")
				return ctrl.Result{}, err
			}

			log.Info("Creating ResourceClaim for WeightClaim")
			if err := r.Create(ctx, resourceClaim); err != nil {
				log.Error(err, "unable to create ResourceClaim")
				return ctrl.Result{}, err
			}
		} else if err != nil {
			log.Error(err, "unable to fetch ResourceClaim")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

// shardResolution contains the resolved shard metadata for a claim.
type shardResolution struct {
	index     int
	path      string
	sizeBytes int64
	err       error
}

// resolveShardIndex determines which shard this claim should be assigned to.
// It uses the explicit ShardIndex from the claim spec if set, otherwise
// falls back to auto-assignment from the claim's name suffix (for JobSet
// compatibility) or returns an error.
func (r *WeightClaimReconciler) resolveShardIndex(
	claim *wpisigk8siov1alpha1.WeightClaim,
	buffer *wpisigk8siov1alpha1.WeightBuffer,
) shardResolution {
	totalShards := buffer.Status.TotalShards
	if totalShards <= 0 {
		return shardResolution{err: fmt.Errorf("WeightBuffer sharding not yet discovered (totalShards=0)")}
	}

	// Determine shard index
	var shardIndex int
	if claim.Spec.ShardIndex != nil {
		shardIndex = *claim.Spec.ShardIndex
	} else {
		// Auto-assignment: try to extract index from pod rank annotations.
		// Check common rank annotations from Job frameworks.
		rankAnnotations := []string{
			"wpi.io/shard-index",               // WPI-specific annotation
			"batch.kubernetes.io/job-completion-index", // K8s Job indexed completions
			"ray.io/rank", // Ray
		}
		found := false
		for _, annotation := range rankAnnotations {
			if val, ok := claim.Annotations[annotation]; ok {
				var idx int
				if _, err := fmt.Sscanf(val, "%d", &idx); err == nil {
					shardIndex = idx
					found = true
					break
				}
			}
		}
		if !found {
			return shardResolution{err: fmt.Errorf(
				"shardIndex not specified and no rank annotation found on claim %q; "+
					"set spec.shardIndex or add annotation wpi.io/shard-index",
				claim.Name)}
		}
	}

	// Validate range
	if shardIndex < 0 || shardIndex >= totalShards {
		return shardResolution{err: fmt.Errorf("shardIndex %d out of range [0, %d)", shardIndex, totalShards)}
	}

	// Look up resolved shard metadata
	if shardIndex < len(buffer.Status.DiscoveredShards) {
		shard := buffer.Status.DiscoveredShards[shardIndex]
		return shardResolution{
			index:     shardIndex,
			path:      shard.Path,
			sizeBytes: shard.SizeBytes,
		}
	}

	// Fallback: shard metadata not yet populated
	return shardResolution{
		index: shardIndex,
		path:  buffer.Spec.SourcePath,
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *WeightClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&wpisigk8siov1alpha1.WeightClaim{}).
		Owns(&resourcev1.ResourceClaim{}).
		Named("weightclaim").
		Complete(r)
}
