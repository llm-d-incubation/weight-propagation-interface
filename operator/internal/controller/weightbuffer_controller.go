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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
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

	// Check if Ready condition already exists
	readyConditionExists := false
	for _, cond := range weightBuffer.Status.Conditions {
		if cond.Type == "Ready" {
			readyConditionExists = true
			break
		}
	}

	if !readyConditionExists {
		weightBuffer.Status.Conditions = append(weightBuffer.Status.Conditions, metav1.Condition{
			Type:               "Ready",
			Status:             metav1.ConditionTrue,
			LastTransitionTime: metav1.Now(),
			Reason:             "BufferCreated",
			Message:            "WeightBuffer has been created and is ready",
		})

		if err := r.Status().Update(ctx, &weightBuffer); err != nil {
			log.Error(err, "unable to update WeightBuffer status")
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *WeightBufferReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&wpisigk8siov1alpha1.WeightBuffer{}).
		Named("weightbuffer").
		Complete(r)
}
