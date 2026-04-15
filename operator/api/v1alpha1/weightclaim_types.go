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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.

// WeightClaimSpec defines the desired state of WeightClaim
type WeightClaimSpec struct {
	// WeightBufferName is the name of the WeightBuffer this claim refers to.
	// +required
	WeightBufferName string `json:"weightBufferName"`
	// ShardIndex specifies which shard of a sharded WeightBuffer this claim
	// is requesting. If the WeightBuffer has sharding and this is not set,
	// the operator auto-assigns based on the pod's rank annotation.
	// +optional
	ShardIndex *int `json:"shardIndex,omitempty"`
}

// WeightClaimStatus defines the observed state of WeightClaim.
type WeightClaimStatus struct {
	// Conditions represent the current state of the WeightClaim resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
	// AssignedShardIndex is the operator-resolved shard index for this claim.
	// Set after validation, whether from explicit spec or auto-assignment.
	// +optional
	AssignedShardIndex *int `json:"assignedShardIndex,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations="api-approved.kubernetes.io=https://github.com/kubernetes/kubernetes/pull/123"

// WeightClaim is the Schema for the weightclaims API
type WeightClaim struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of WeightClaim
	// +required
	Spec WeightClaimSpec `json:"spec"`

	// status defines the observed state of WeightClaim
	// +optional
	Status WeightClaimStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// WeightClaimList contains a list of WeightClaim
type WeightClaimList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []WeightClaim `json:"items"`
}

func init() {
	SchemeBuilder.Register(&WeightClaim{}, &WeightClaimList{})
}
