package main

import (
	"context"
	"fmt"
	"log"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	drav1 "k8s.io/kubelet/pkg/apis/dra/v1"
	wpipb "wpi.sig.k8s.io/operator/pkg/wpi"
)

func (p *wpiPlugin) NodeUnprepareResources(ctx context.Context, req *drav1.NodeUnprepareResourcesRequest) (*drav1.NodeUnprepareResourcesResponse, error) {
// I will just draft the logic here.
