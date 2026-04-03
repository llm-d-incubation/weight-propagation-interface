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

// WeightBufferSpec defines the desired state of WeightBuffer
type WeightBufferSpec struct {
	// Capacity is the total size of the weight buffer (e.g., "10Gi").
	// +optional
	Capacity string `json:"capacity,omitempty"`
	// SourcePath is the URI or local path to load the weights from
	// +optional
	SourcePath string `json:"sourcePath,omitempty"`
}

// WeightBufferStatus defines the observed state of WeightBuffer.
type WeightBufferStatus struct {
	// Conditions represent the current state of the WeightBuffer resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:metadata:annotations="api-approved.kubernetes.io=https://github.com/kubernetes/kubernetes/pull/123"

// WeightBuffer is the Schema for the weightbuffers API
type WeightBuffer struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is a standard object metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitzero"`

	// spec defines the desired state of WeightBuffer
	// +required
	Spec WeightBufferSpec `json:"spec"`

	// status defines the observed state of WeightBuffer
	// +optional
	Status WeightBufferStatus `json:"status,omitzero"`
}

// +kubebuilder:object:root=true

// WeightBufferList contains a list of WeightBuffer
type WeightBufferList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitzero"`
	Items           []WeightBuffer `json:"items"`
}

func init() {
	SchemeBuilder.Register(&WeightBuffer{}, &WeightBufferList{})
}
