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

	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

// DRAReconciler reconciles a ResourceClaim object
type DRAReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims/status,verbs=get;update;patch

// Reconcile is part of the main kubernetes reconciliation loop
func (r *DRAReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var rc resourcev1.ResourceClaim
	if err := r.Get(ctx, req.NamespacedName, &rc); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		log.Error(err, "unable to fetch ResourceClaim")
		return ctrl.Result{}, err
	}

	// Skip if it is not for our driver (we check DeviceClassName)
	// We'll process if any request asks for "wpi-device-class"
	isWPI := false
	if rc.Spec.Devices.Requests != nil {
		for _, req := range rc.Spec.Devices.Requests {
			if req.Exactly != nil && req.Exactly.DeviceClassName == "wpi-device-class" {
				isWPI = true
				break
			}
		}
	}

	if !isWPI {
		return ctrl.Result{}, nil
	}

	// If already allocated with devices, do nothing
	if rc.Status.Allocation != nil && len(rc.Status.Allocation.Devices.Results) > 0 {
		return ctrl.Result{}, nil
	}

	// Native Kubernetes Scheduler (1.31+) will handle the allocation natively
	// as long as ResourceSlices are properly published by the driver DaemonSet.
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *DRAReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&resourcev1.ResourceClaim{}).
		Named("dracontroller").
		Complete(r)
}
