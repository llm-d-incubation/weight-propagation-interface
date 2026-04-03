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

	resourcev1 "k8s.io/api/resource/v1"
	wpisigk8siov1alpha1 "wpi.sig.k8s.io/operator/api/v1alpha1"
)

// WeightClaimReconciler reconciles a WeightClaim object
type WeightClaimReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=wpi.sig.k8s.io,resources=weightclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=wpi.sig.k8s.io,resources=weightclaims/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=wpi.sig.k8s.io,resources=weightclaims/finalizers,verbs=update
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

// SetupWithManager sets up the controller with the Manager.
func (r *WeightClaimReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&wpisigk8siov1alpha1.WeightClaim{}).
		Owns(&resourcev1.ResourceClaim{}).
		Named("weightclaim").
		Complete(r)
}
