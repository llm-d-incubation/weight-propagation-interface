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

// ShardingStrategy defines how a model is partitioned across GPUs.
// +kubebuilder:validation:Enum=TensorParallel;ExpertParallel;PipelineParallel;Custom
type ShardingStrategy string

const (
	// TensorParallel splits each tensor across GPUs along a chosen dimension.
	TensorParallel ShardingStrategy = "TensorParallel"
	// ExpertParallel distributes MoE experts across GPUs.
	ExpertParallel ShardingStrategy = "ExpertParallel"
	// PipelineParallel assigns sequential layers to different GPUs.
	PipelineParallel ShardingStrategy = "PipelineParallel"
	// CustomSharding allows user-defined byte-range partitioning.
	CustomSharding ShardingStrategy = "Custom"
)

// ShardFile maps a shard index to its source file on shared storage.
type ShardFile struct {
	// Index is the shard number (0-indexed).
	// +required
	Index int `json:"index"`
	// Path to the shard file on shared storage.
	// +required
	Path string `json:"path"`
	// SizeBytes is the size of this specific shard in bytes.
	// +optional
	SizeBytes int64 `json:"sizeBytes,omitempty"`
}

// ShardingSpec defines how a WeightBuffer is partitioned into shards
// for distribution across multiple GPUs or nodes.
type ShardingSpec struct {
	// Strategy is the parallelism pattern used to split the model.
	// +required
	Strategy ShardingStrategy `json:"strategy"`
	// NumShards is the total number of shards the model is split into.
	// +required
	NumShards int `json:"numShards"`
	// ShardFiles is an explicit mapping of shard index to file.
	// If omitted, the operator uses FilePattern for auto-discovery.
	// +optional
	ShardFiles []ShardFile `json:"shardFiles,omitempty"`
	// FilePattern is a glob pattern for auto-discovering shard files
	// within SourcePath (e.g., "model-*-of-*.safetensors").
	// Mutually exclusive with ShardFiles.
	// +optional
	FilePattern string `json:"filePattern,omitempty"`
}

// WeightBufferSpec defines the desired state of WeightBuffer
type WeightBufferSpec struct {
	// Capacity is the total size of the weight buffer (e.g., "10Gi").
	// +optional
	Capacity string `json:"capacity,omitempty"`
	// SourcePath is the URI or local path to load the weights from.
	// For sharded models, this is the directory containing shard files.
	// +optional
	SourcePath string `json:"sourcePath,omitempty"`
	// Sharding defines how this buffer is partitioned across GPUs.
	// If nil, the buffer is treated as a single monolithic block.
	// +optional
	Sharding *ShardingSpec `json:"sharding,omitempty"`
}

// DiscoveredShard represents a resolved shard after discovery.
type DiscoveredShard struct {
	// Index is the shard number (0-indexed).
	Index int `json:"index"`
	// Path is the resolved path to the shard file.
	Path string `json:"path"`
	// SizeBytes is the resolved size of this shard.
	SizeBytes int64 `json:"sizeBytes"`
}

// WeightBufferStatus defines the observed state of WeightBuffer.
type WeightBufferStatus struct {
	// Conditions represent the current state of the WeightBuffer resource.
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
	// TotalShards is the resolved total number of shards.
	// 0 or absent means the buffer is not sharded.
	// +optional
	TotalShards int `json:"totalShards,omitempty"`
	// DiscoveredShards contains the resolved metadata for each shard
	// after the operator has performed shard discovery.
	// +optional
	DiscoveredShards []DiscoveredShard `json:"discoveredShards,omitempty"`
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
