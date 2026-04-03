package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	drav1 "k8s.io/kubelet/pkg/apis/dra/v1"
	pluginregv1 "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
	wpipb "wpi.sig.k8s.io/operator/pkg/wpi"
)

const (
	pluginName     = "wpi.sig.k8s.io"
	pluginEndpoint = "/var/lib/kubelet/plugins/wpi.sig.k8s.io/dra.sock"
	driverEndpoint = "unix:///run/wpi/sockets/wpi-grpc.sock" // Connection to python wpi-driver
	registrySocket = "/var/lib/kubelet/plugins_registry/wpi.sig.k8s.io-dra.sock"
)

// wpiPlugin implements drav1.DRAPluginServer and pluginregv1.RegistrationServer
type wpiPlugin struct {
	pluginregv1.UnimplementedRegistrationServer
	drav1.UnimplementedDRAPluginServer
}

func (p *wpiPlugin) GetInfo(ctx context.Context, req *pluginregv1.InfoRequest) (*pluginregv1.PluginInfo, error) {
	log.Println("GetInfo called by plugin registry")
	return &pluginregv1.PluginInfo{
		Type:              pluginregv1.DRAPlugin,
		Name:              pluginName,
		Endpoint:          pluginEndpoint,
		SupportedVersions: []string{"v1.DRAPlugin", "v1beta1.DRAPlugin"},
	}, nil
}

func (p *wpiPlugin) NotifyRegistrationStatus(ctx context.Context, req *pluginregv1.RegistrationStatus) (*pluginregv1.RegistrationStatusResponse, error) {
	log.Printf("Registration status: %v", req.PluginRegistered)
	if !req.PluginRegistered {
		log.Printf("Registration failed: %s", req.Error)
	}
	return &pluginregv1.RegistrationStatusResponse{}, nil
}

func (p *wpiPlugin) NodePrepareResources(ctx context.Context, req *drav1.NodePrepareResourcesRequest) (*drav1.NodePrepareResourcesResponse, error) {
	log.Printf("NodePrepareResources called for %d claims", len(req.Claims))

	// Create connection to the local python driver
	conn, err := grpc.NewClient(driverEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Failed to connect to local wpi-driver: %v", err)
		return nil, fmt.Errorf("failed to connect to wpi-driver: %v", err)
	}
	defer conn.Close()

	// Initialize Kubernetes client to read CRDs
	k8sConfig, err := rest.InClusterConfig()
	if err != nil {
		log.Printf("Failed to create in-cluster config: %v", err)
		return nil, fmt.Errorf("failed to create in-cluster config: %v", err)
	}

	// Create dynamic client to fetch WeightClaim/WeightBuffer
	dynClient, err := dynamic.NewForConfig(k8sConfig)
	if err != nil {
		log.Printf("Failed to create dynamic client: %v", err)
		return nil, fmt.Errorf("failed to create dynamic client: %v", err)
	}

	wcGVR := schema.GroupVersionResource{Group: "wpi.sig.k8s.io", Version: "v1alpha1", Resource: "weightclaims"}
	wbGVR := schema.GroupVersionResource{Group: "wpi.sig.k8s.io", Version: "v1alpha1", Resource: "weightbuffers"}

	client := wpipb.NewNodeServiceClient(conn)
	for _, claim := range req.Claims {
		claimUID := string(claim.Uid)
		namespace := claim.Namespace
		name := claim.Name

		log.Printf("Processing claim UID: %s, Namespace: %s, Name: %s", claimUID, namespace, name)

		var sourcePath string
		var sizeBytes uint64
		var bufferName string

		rcGVR := schema.GroupVersionResource{Group: "resource.k8s.io", Version: "v1", Resource: "resourceclaims"}
		
		claimsList, err := dynClient.Resource(rcGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
		var realClaimName string
		if err == nil {
			for _, item := range claimsList.Items {
				if string(item.GetUID()) == claimUID {
					realClaimName = item.GetName()
					if _, found, _ := unstructured.NestedString(item.Object, "metadata", "ownerReferences", "0", "name"); found {
						// Sometimes the custom controller makes the name of the ResourceClaim the UUID, but it's owned by WeightClaim
						// Actually, the Custom Controller creates a ResourceClaim where its Name == WeightClaim name (e.g. wc-test)
						log.Printf("Found matching ResourceClaim: %s", realClaimName)
					}
					break
				}
			}
		} else {
			log.Printf("Failed to list ResourceClaims: %v", err)
		}

		if realClaimName == "" {
			log.Printf("Warning: Could not find ResourceClaim with UID %s in namespace %s. Trying name=%s as fallback.", claimUID, namespace, name)
			realClaimName = name
		}

		// 1. Fetch the WeightClaim to find the WeightBufferName using the realClaimName
		wcUnstructured, err := dynClient.Resource(wcGVR).Namespace(namespace).Get(ctx, realClaimName, metav1.GetOptions{})
		if err == nil {
			var found bool
			bufferName, found, _ = unstructured.NestedString(wcUnstructured.Object, "spec", "weightBufferName")
			if found && bufferName != "" {
				// 2. Fetch the WeightBuffer to get Capacity and SourcePath
				wbUnstructured, err := dynClient.Resource(wbGVR).Namespace(namespace).Get(ctx, bufferName, metav1.GetOptions{})
				if err == nil {
					sourcePath, _, _ = unstructured.NestedString(wbUnstructured.Object, "spec", "sourcePath")
					capacityStr, _, _ := unstructured.NestedString(wbUnstructured.Object, "spec", "capacity")

					// Parse capacity (e.g. "5Gi", "10GiB") into bytes
					if capacityStr != "" {
						qty, err := resource.ParseQuantity(capacityStr)
						if err == nil {
							sizeBytes = uint64(qty.Value())
						} else {
							log.Printf("Failed to parse capacity '%s': %v", capacityStr, err)
						}
					}
					log.Printf("Found WeightBuffer %s: SourcePath=%s, SizeBytes=%d", bufferName, sourcePath, sizeBytes)

					totalVRAM, vramErr := getTotalVRAM()
					if vramErr == nil && sizeBytes > totalVRAM {
						errStr := fmt.Sprintf("requested WeightBuffer capacity %d bytes exceeds total node VRAM %d bytes", sizeBytes, totalVRAM)
						log.Printf(errStr)
						return nil, fmt.Errorf(errStr)
					}
				} else {
					log.Printf("Failed to fetch WeightBuffer %s: %v", bufferName, err)
				}
			}
		} else {
			log.Printf("Failed to fetch WeightClaim %s: %v", realClaimName, err)
		}

		if bufferName == "" {
			log.Printf("Failed to find valid WeightBufferName for WeightClaim %s. Relying on defaults if possible.", realClaimName)
		}

		_, err = client.NodeStageWeight(ctx, &wpipb.NodeStageWeightRequest{
			ClaimId:    realClaimName,
			BufferId:   bufferName,
			SourcePath: sourcePath,
			SizeBytes:  sizeBytes,
		})
		if err != nil {
			log.Printf("Failed to call NodeStageWeight for claim %s: %v", claim.Uid, err)
			return nil, fmt.Errorf("failed to process claim %s: %v", claim.Uid, err)
		}
	}

	log.Printf("Successfully proxy-called NodeStageWeight to python driver for claims")

	res := &drav1.NodePrepareResourcesResponse{
		Claims: make(map[string]*drav1.NodePrepareResourceResponse),
	}
	for _, claim := range req.Claims {
		res.Claims[claim.Uid] = &drav1.NodePrepareResourceResponse{}
	}

	return res, nil
}

func (p *wpiPlugin) NodeUnprepareResources(ctx context.Context, req *drav1.NodeUnprepareResourcesRequest) (*drav1.NodeUnprepareResourcesResponse, error) {
	log.Printf("NodeUnprepareResources called for %d claims", len(req.Claims))

	conn, err := grpc.NewClient(driverEndpoint, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Printf("Failed to connect to local wpi-driver: %v", err)
		return nil, fmt.Errorf("failed to connect to wpi-driver: %v", err)
	}
	defer conn.Close()

	k8sConfig, err := rest.InClusterConfig()
	if err != nil {
		log.Printf("Failed to create in-cluster config: %v", err)
		return nil, fmt.Errorf("failed to create in-cluster config: %v", err)
	}

	dynClient, err := dynamic.NewForConfig(k8sConfig)
	if err != nil {
		log.Printf("Failed to create dynamic client: %v", err)
		return nil, fmt.Errorf("failed to create dynamic client: %v", err)
	}

	client := wpipb.NewNodeServiceClient(conn)

	res := &drav1.NodeUnprepareResourcesResponse{
		Claims: make(map[string]*drav1.NodeUnprepareResourceResponse),
	}

	for _, claim := range req.Claims {
		claimUID := string(claim.Uid)
		namespace := claim.Namespace
		name := claim.Name
		log.Printf("Unpreparing claim UID: %s, Namespace: %s, Name: %s", claimUID, namespace, name)

		rcGVR := schema.GroupVersionResource{Group: "resource.k8s.io", Version: "v1", Resource: "resourceclaims"}
		var realClaimName string
		claimsList, err := dynClient.Resource(rcGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
		if err == nil {
			for _, item := range claimsList.Items {
				if string(item.GetUID()) == claimUID {
					realClaimName = item.GetName()
					if _, found, _ := unstructured.NestedString(item.Object, "metadata", "ownerReferences", "0", "name"); found {
						log.Printf("Found matching ResourceClaim: %s", realClaimName)
					}
					break
				}
			}
		}

		if realClaimName == "" {
			realClaimName = name
		}

		// Do not query for WeightClaim since it might have been deleted by cascade
		_, err = client.NodeUnstageWeight(ctx, &wpipb.NodeUnstageWeightRequest{
			ClaimId: realClaimName,
		})
		if err != nil {
			log.Printf("Failed to call NodeUnstageWeight for claim %s: %v", claim.Uid, err)
		} else {
			log.Printf("Successfully proxy-called NodeUnstageWeight for claim %s", realClaimName)
		}

		res.Claims[claim.Uid] = &drav1.NodeUnprepareResourceResponse{}
	}
	return res, nil
}

func main() {
	log.Println("Starting WPI DRA Kubelet Plugin...")

	if err := os.MkdirAll(filepath.Dir(pluginEndpoint), 0755); err != nil {
		log.Fatalf("Failed to create plugin dir: %v", err)
	}
	// Cleanup existing sockets
	os.Remove(pluginEndpoint)
	os.Remove(registrySocket)

	pluginListener, err := net.Listen("unix", pluginEndpoint)
	if err != nil {
		log.Fatalf("Failed to listen on plugin endpoint: %v", err)
	}

	registryListener, err := net.Listen("unix", registrySocket)
	if err != nil {
		log.Fatalf("Failed to listen on registry socket: %v", err)
	}

	plugin := &wpiPlugin{}

	pluginServer := grpc.NewServer()
	drav1.RegisterDRAPluginServer(pluginServer, plugin)
	pluginregv1.RegisterRegistrationServer(pluginServer, plugin)

	registryServer := grpc.NewServer()
	pluginregv1.RegisterRegistrationServer(registryServer, plugin)

	nodeName := os.Getenv("NODE_NAME")
	if nodeName != "" {
		k8sConfig, err := rest.InClusterConfig()
		if err == nil {
			if dynClient, err := dynamic.NewForConfig(k8sConfig); err == nil {
				go publishResourceSliceLoop(context.Background(), dynClient, nodeName)
			}
		} else {
			log.Printf("Failed to get config for ResourceSlice publisher: %v", err)
		}
	} else {
		log.Println("NODE_NAME env var not set, skipping ResourceSlice publishing")
	}

	go func() {
		log.Printf("Serving plugin at %s", pluginEndpoint)
		if err := pluginServer.Serve(pluginListener); err != nil {
			log.Fatalf("Plugin server failed: %v", err)
		}
	}()

	go func() {
		log.Printf("Serving registry at %s", registrySocket)
		if err := registryServer.Serve(registryListener); err != nil {
			log.Fatalf("Registry server failed: %v", err)
		}
	}()

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)
	<-c

	log.Println("Shutting down...")
	pluginServer.Stop()
	registryServer.Stop()
}
