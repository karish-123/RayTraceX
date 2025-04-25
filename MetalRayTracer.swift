import Metal
import MetalKit
import simd
import SwiftUI

// MARK: - Data Structures

struct Sphere {
    var center: SIMD3<Float>
    var radius: Float
    var material: Int32
}

struct Ray {
    var origin: SIMD3<Float>
    var direction: SIMD3<Float>
}

struct Camera {
    var position: SIMD3<Float>
    var lowerLeftCorner: SIMD3<Float>
    var horizontal: SIMD3<Float>
    var vertical: SIMD3<Float>
}

struct Material {
    var albedo: SIMD3<Float>
    var type: Int32
    var fuzziness: Float
    var refractiveIndex: Float
}

struct Constants {
    var width: UInt32
    var height: UInt32
    var sphereCount: UInt32
    var materialCount: UInt32
    var frameCount: UInt32
}

// MARK: - BVH Structures

struct AABB {
    var min: SIMD3<Float>
    var max: SIMD3<Float>
    
    static func forSphere(center: SIMD3<Float>, radius: Float) -> AABB {
        let min = center - SIMD3<Float>(radius, radius, radius)
        let max = center + SIMD3<Float>(radius, radius, radius)
        return AABB(min: min, max: max)
    }
    
    static func union(_ box1: AABB, _ box2: AABB) -> AABB {
        let min = SIMD3<Float>(
            Swift.min(box1.min.x, box2.min.x),
            Swift.min(box1.min.y, box2.min.y),
            Swift.min(box1.min.z, box2.min.z)
        )
        
        let max = SIMD3<Float>(
            Swift.max(box1.max.x, box2.max.x),
            Swift.max(box1.max.y, box2.max.y),
            Swift.max(box1.max.z, box2.max.z)
        )
        
        return AABB(min: min, max: max)
    }
}

struct BVHNode {
    var min: SIMD3<Float>           // Bounding box minimum
    var leftFirst: Int32            // Index of left child or first primitive
    var max: SIMD3<Float>           // Bounding box maximum
    var primitiveCount: Int32       // Number of primitives (0 for interior nodes)
}

// MARK: - BVH Builder

class BVHBuilder {
    // Builds a BVH for the given spheres
    func build(spheres: [Sphere]) -> (nodes: [BVHNode], indices: [Int32]) {
        let startTime = CACurrentMediaTime()
        print("Building BVH on CPU...")
        
        // Create primitive indices array
        let primitiveIndices = Array(0..<Int32(spheres.count))
        
        // Create an array to store centroids
        var centroids = [SIMD3<Float>](repeating: .zero, count: spheres.count)
        for i in 0..<spheres.count {
            centroids[i] = spheres[i].center
        }
        
        // Build the BVH tree recursively
        var nodes = [BVHNode]()
        var orderedIndices = [Int32]()
        
        // Start with empty root node
        let rootAABB = calculateBounds(spheres: spheres, indices: primitiveIndices)
        nodes.append(BVHNode(min: rootAABB.min, leftFirst: 0, max: rootAABB.max, primitiveCount: 0))
        
        // Build tree recursively
        buildRecursive(
            spheres: spheres,
            indices: primitiveIndices,
            centroids: centroids,
            nodeIndex: 0,
            nodes: &nodes,
            orderedIndices: &orderedIndices
        )
        
        let endTime = CACurrentMediaTime()
        print("BVH built with \(nodes.count) nodes in \(endTime - startTime) seconds")
        
        return (nodes, orderedIndices)
    }
    
    // Helper function to calculate AABB for a set of spheres
    private func calculateBounds(spheres: [Sphere], indices: [Int32]) -> AABB {
        guard !indices.isEmpty else {
            return AABB(min: .zero, max: .zero)
        }
        
        let firstSphere = spheres[Int(indices[0])]
        var bounds = AABB.forSphere(center: firstSphere.center, radius: firstSphere.radius)
        
        for i in 1..<indices.count {
            let sphere = spheres[Int(indices[i])]
            let sphereBounds = AABB.forSphere(center: sphere.center, radius: sphere.radius)
            bounds = AABB.union(bounds, sphereBounds)
        }
        
        return bounds
    }
    
    // Recursive function to build the BVH
    private func buildRecursive(
        spheres: [Sphere],
        indices: [Int32],
        centroids: [SIMD3<Float>],
        nodeIndex: Int,
        nodes: inout [BVHNode],
        orderedIndices: inout [Int32]
    ) {
        let primitiveCount = indices.count
        
        // Leaf node case - store primitives directly
        if primitiveCount <= 4 {
            let firstIndex = Int32(orderedIndices.count)
            orderedIndices.append(contentsOf: indices)
            
            // Update the node to be a leaf
            nodes[nodeIndex].leftFirst = firstIndex
            nodes[nodeIndex].primitiveCount = Int32(primitiveCount)
            return
        }
        
        // Find the largest axis of the bounding box
        let bounds = AABB(min: nodes[nodeIndex].min, max: nodes[nodeIndex].max)
        let extent = bounds.max - bounds.min
        let axis = extent.x > extent.y ? (extent.x > extent.z ? 0 : 2) : (extent.y > extent.z ? 1 : 2)
        
        // Sort indices based on centroids along the chosen axis
        let sortedIndices = indices.sorted {
            return centroids[Int($0)][axis] < centroids[Int($1)][axis]
        }
        
        // Split the primitives into two groups
        let mid = primitiveCount / 2
        let leftIndices = Array(sortedIndices[0..<mid])
        let rightIndices = Array(sortedIndices[mid..<primitiveCount])
        
        // Create child nodes
        let leftBounds = calculateBounds(spheres: spheres, indices: leftIndices)
        let rightBounds = calculateBounds(spheres: spheres, indices: rightIndices)
        
        // Add left node to the array
        let leftIndex = nodes.count
        nodes.append(BVHNode(
            min: leftBounds.min,
            leftFirst: 0, // Will be updated when processing this node
            max: leftBounds.max,
            primitiveCount: 0
        ))
        
        // Add right node to the array
        let rightIndex = nodes.count
        nodes.append(BVHNode(
            min: rightBounds.min,
            leftFirst: 0, // Will be updated when processing this node
            max: rightBounds.max,
            primitiveCount: 0
        ))
        
        // Update current node to point to its children
        nodes[nodeIndex].leftFirst = Int32(leftIndex)
        
        // Recursively build the child nodes
        buildRecursive(
            spheres: spheres,
            indices: leftIndices,
            centroids: centroids,
            nodeIndex: leftIndex,
            nodes: &nodes,
            orderedIndices: &orderedIndices
        )
        
        buildRecursive(
            spheres: spheres,
            indices: rightIndices,
            centroids: centroids,
            nodeIndex: rightIndex,
            nodes: &nodes,
            orderedIndices: &orderedIndices
        )
    }
}

// MARK: - Metal Ray Tracer Implementation

class MetalRayTracer: NSObject, MTKViewDelegate {
    // Metal properties
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var computePipeline: MTLComputePipelineState!
    var resultTexture: MTLTexture!
    
    // Scene data
    var spheres: [Sphere] = []
    var materials: [Material] = []
    var camera: Camera!
    
    // Buffers for GPU
    var sphereBuffer: MTLBuffer!
    var materialBuffer: MTLBuffer!
    var cameraBuffer: MTLBuffer!
    var constantsBuffer: MTLBuffer!
    
    // BVH data
    var bvhNodes: [BVHNode] = []
    var bvhIndices: [Int32] = []
    var bvhNodeBuffer: MTLBuffer!
    var bvhIndexBuffer: MTLBuffer!
    var useBVH: Bool = true
    
    // Rendering properties
    var width: Int = 800
    var height: Int = 600
    var constants = Constants(width: 800, height: 600, sphereCount: 0, materialCount: 0, frameCount: 0)
    
    // Debug and performance properties
    var lastFrameTime: CFTimeInterval = 0
    var frameRateLabel: NSTextField?
    var benchmarkMode: Bool = false
    private var benchmarkStep = 0
    private var benchmarkStartTime: CFTimeInterval = 0
    private var frameCounter = 0
    
    // MARK: Setup and Initialization
    
    func setup(view: MTKView) {
        print("Setting up Metal Ray Tracer")
        
        // Initialize Metal
        device = MTLCreateSystemDefaultDevice()
        guard device != nil else {
            fatalError("Metal is not supported on this device")
        }
        
        view.device = device
        view.delegate = self
        view.framebufferOnly = false
        view.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        
        commandQueue = device.makeCommandQueue()
        
        // Create shader with function constants for BVH
        let library = device.makeDefaultLibrary()
        
        // Define function constants
        let functionConstants = MTLFunctionConstantValues()
        var useBVHValue = useBVH
        functionConstants.setConstantValue(&useBVHValue, type: .bool, index: 0)
        
        do {
            let rayTracingFunction = try library?.makeFunction(name: "rayTracing", constantValues: functionConstants)
            computePipeline = try device.makeComputePipelineState(function: rayTracingFunction!)
        } catch {
            print("Error creating ray tracing function with constants: \(error)")
            // Fallback to standard function without constants
            let rayTracingFunction = library?.makeFunction(name: "rayTracing")
            do {
                computePipeline = try device.makeComputePipelineState(function: rayTracingFunction!)
            } catch {
                fatalError("Failed to create compute pipeline: \(error)")
            }
        }
        
        // Create scene and build BVH
        createScene()
        
        // Create buffers
        updateBuffers()
        
        // Create result texture
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        resultTexture = device.makeTexture(descriptor: textureDescriptor)
        
        print("Setup complete. Created scene with \(spheres.count) spheres and \(materials.count) materials")
        
        // Add UI controls
        addControls(to: view)
    }
    
    func addControls(to view: MTKView) {
        // BVH toggle button
        let toggleButton = NSButton(frame: NSRect(x: 10, y: 40, width: 100, height: 30))
        toggleButton.title = "Toggle BVH"
        toggleButton.bezelStyle = .rounded
        toggleButton.target = self
        toggleButton.action = #selector(toggleBVH)
        view.addSubview(toggleButton)
        
        // Benchmark button
        let benchmarkButton = NSButton(frame: NSRect(x: 120, y: 40, width: 100, height: 30))
        benchmarkButton.title = "Benchmark"
        benchmarkButton.bezelStyle = .rounded
        benchmarkButton.target = self
        benchmarkButton.action = #selector(runBenchmark)
        view.addSubview(benchmarkButton)
    }
    
    @objc func toggleBVH() {
        useBVH = !useBVH
        print("BVH is now \(useBVH ? "enabled" : "disabled")")
        
        // Recreate pipeline with new function constant
        let library = device.makeDefaultLibrary()
        let functionConstants = MTLFunctionConstantValues()
        var useBVHValue = useBVH
        functionConstants.setConstantValue(&useBVHValue, type: .bool, index: 0)
        
        do {
            let rayTracingFunction = try library?.makeFunction(name: "rayTracing", constantValues: functionConstants)
            computePipeline = try device.makeComputePipelineState(function: rayTracingFunction!)
        } catch {
            print("Error recreating pipeline: \(error)")
        }
        
        // Reset frame counter to restart progressive rendering
        constants.frameCount = 0
    }
    
    @objc func runBenchmark() {
        benchmarkMode = true
        constants.frameCount = 0
        
        // First benchmark with BVH disabled
        if useBVH {
            useBVH = false
            toggleBVH()
        }
        
        print("Starting benchmark with BVH disabled...")
        // The actual benchmark will run for 100 frames, then switch to BVH enabled
        // This is handled in the draw method
    }
    
    // MARK: Scene and BVH Creation
    
    func createScene() {
        print("Creating scene...")
        
        // Create materials
        createMaterials()
        
        // Create spheres
        createSpheres()
        
        // Update constants
        constants.sphereCount = UInt32(spheres.count)
        constants.materialCount = UInt32(materials.count)
        
        // Build BVH (utilizing CPU)
        buildBVH()
        
        // Setup camera
        setupCamera()
    }
    
    func createMaterials() {
        materials = [
            // Ground material - beige diffuse
            Material(albedo: SIMD3<Float>(0.76, 0.69, 0.57), type: 0, fuzziness: 0, refractiveIndex: 1.0),
            
            // Diffuse materials with different colors
            Material(albedo: SIMD3<Float>(0.9, 0.2, 0.2), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Red
            Material(albedo: SIMD3<Float>(0.2, 0.9, 0.2), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Green
            Material(albedo: SIMD3<Float>(0.2, 0.2, 0.9), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Blue
            Material(albedo: SIMD3<Float>(0.9, 0.9, 0.2), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Yellow
            Material(albedo: SIMD3<Float>(0.9, 0.5, 0.2), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Orange
            Material(albedo: SIMD3<Float>(0.9, 0.3, 0.9), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Pink
            Material(albedo: SIMD3<Float>(0.3, 0.9, 0.9), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Cyan
            Material(albedo: SIMD3<Float>(0.5, 0.3, 0.9), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Purple
            Material(albedo: SIMD3<Float>(0.6, 0.4, 0.2), type: 0, fuzziness: 0, refractiveIndex: 1.0),  // Brown
            
            // Perfect mirror (highly reflective metal)
            Material(albedo: SIMD3<Float>(0.95, 0.95, 0.95), type: 1, fuzziness: 0.0, refractiveIndex: 1.0),
            
            // Standard glass
            Material(albedo: SIMD3<Float>(1.0, 1.0, 1.0), type: 2, fuzziness: 0.0, refractiveIndex: 1.5),
            
            // Deep blue diffuse for the third main sphere
            Material(albedo: SIMD3<Float>(0.1, 0.2, 0.8), type: 0, fuzziness: 0, refractiveIndex: 1.0)
        ]
        
        // Create additional random materials
        for _ in 0..<20 {
            let materialType = Int32(arc4random_uniform(3))  // 0, 1, or 2
            
            if materialType == 0 {  // Diffuse
                materials.append(Material(
                    albedo: SIMD3<Float>(
                        Float.random(in: 0.3...1.0),
                        Float.random(in: 0.3...1.0),
                        Float.random(in: 0.3...1.0)
                    ),
                    type: 0,
                    fuzziness: 0,
                    refractiveIndex: 1.0
                ))
            } else if materialType == 1 {  // Metal
                materials.append(Material(
                    albedo: SIMD3<Float>(
                        Float.random(in: 0.5...1.0),
                        Float.random(in: 0.5...1.0),
                        Float.random(in: 0.5...1.0)
                    ),
                    type: 1,
                    fuzziness: Float.random(in: 0.0...0.5),
                    refractiveIndex: 1.0
                ))
            } else {  // Glass
                materials.append(Material(
                    albedo: SIMD3<Float>(1.0, 1.0, 1.0),
                    type: 2,
                    fuzziness: 0,
                    refractiveIndex: Float.random(in: 1.3...2.0)
                ))
            }
        }
    }
    
    func createSpheres() {
        spheres = []
        materials = []

        // 0 - Ground material (lambertian)
        materials.append(Material(albedo: SIMD3<Float>(0.5, 0.5, 0.5), type: 0, fuzziness: 0, refractiveIndex: 1.0))
        spheres.append(Sphere(center: SIMD3<Float>(0, -1000, 0), radius: 1000, material: 0))

        // Grid of small random spheres
        for a in -11..<11 {
            for b in -11..<11 {
                let chooseMat = Float.random(in: 0...1)
                let center = SIMD3<Float>(
                    Float(a) + 0.9 * Float.random(in: 0...1),
                    0.2,
                    Float(b) + 0.9 * Float.random(in: 0...1)
                )

                if length(center - SIMD3<Float>(4, 0.2, 0)) > 0.9 {
                    var materialIndex: Int32 = 0

                    if chooseMat < 0.8 {
                        // Lambertian
                        let albedo = SIMD3<Float>(
                            Float.random(in: 0...1),
                            Float.random(in: 0...1),
                            Float.random(in: 0...1)
                        ) * SIMD3<Float>(
                            Float.random(in: 0...1),
                            Float.random(in: 0...1),
                            Float.random(in: 0...1)
                        )
                        materials.append(Material(albedo: albedo, type: 0, fuzziness: 0, refractiveIndex: 1.0))
                        materialIndex = Int32(materials.count - 1)
                    } else if chooseMat < 0.95 {
                        // Metal
                        let albedo = SIMD3<Float>(
                            Float.random(in: 0.5...1),
                            Float.random(in: 0.5...1),
                            Float.random(in: 0.5...1)
                        )
                        let fuzz = Float.random(in: 0...0.5)
                        materials.append(Material(albedo: albedo, type: 1, fuzziness: fuzz, refractiveIndex: 1.0))
                        materialIndex = Int32(materials.count - 1)
                    } else {
                        // Glass
                        materials.append(Material(albedo: SIMD3<Float>(1, 1, 1), type: 2, fuzziness: 0, refractiveIndex: 1.5))
                        materialIndex = Int32(materials.count - 1)
                    }

                    spheres.append(Sphere(center: center, radius: 0.2, material: materialIndex))
                }
            }
        }

        // Main large glass sphere in center
        let glassMat = Int32(materials.count)
        materials.append(Material(albedo: SIMD3<Float>(1, 1, 1), type: 2, fuzziness: 0, refractiveIndex: 1.5))
        spheres.append(Sphere(center: SIMD3<Float>(0, 1, 0), radius: 1.0, material: glassMat))

        // Main large diffuse sphere on the left
        let lambertMat = Int32(materials.count)
        materials.append(Material(albedo: SIMD3<Float>(0.4, 0.2, 0.1), type: 0, fuzziness: 0, refractiveIndex: 1.0))
        spheres.append(Sphere(center: SIMD3<Float>(-4, 1, 0), radius: 1.0, material: lambertMat))

        // Main large metal sphere on the right
        let metalMat = Int32(materials.count)
        materials.append(Material(albedo: SIMD3<Float>(0.7, 0.6, 0.5), type: 1, fuzziness: 0.0, refractiveIndex: 1.0))
        spheres.append(Sphere(center: SIMD3<Float>(4, 1, 0), radius: 1.0, material: metalMat))

        // Update constants
        constants.sphereCount = UInt32(spheres.count)
        constants.materialCount = UInt32(materials.count)

        print("Created scene with \(spheres.count) spheres and \(materials.count) materials")
    }

    
    func buildBVH() {
        let builder = BVHBuilder()
        let bvhData = builder.build(spheres: spheres)
        bvhNodes = bvhData.nodes
        bvhIndices = bvhData.indices
        
        // Create Metal buffers for BVH data
        updateBVHBuffers()
    }
    
    func setupCamera() {
        // Position camera for a good view of the scene
        let lookFrom = SIMD3<Float>(-13, 2, -3)
        let lookAt = SIMD3<Float>(0, 0, 0)
        let up = SIMD3<Float>(0, 1, 0)
        let fov: Float = 40
        let aspect: Float = Float(width) / Float(height)
        let focusDistance: Float = length(lookFrom - lookAt)
        
        let theta = fov * Float.pi / 180
        let halfHeight = tan(theta / 2)
        let halfWidth = aspect * halfHeight
        
        let w = normalize(lookFrom - lookAt)
        let u = normalize(cross(up, w))
        let v = cross(w, u)
        
        camera = Camera(
            position: lookFrom,
            lowerLeftCorner: lookFrom - halfWidth * focusDistance * u - halfHeight * focusDistance * v - focusDistance * w,
            horizontal: 2 * halfWidth * focusDistance * u,
            vertical: 2 * halfHeight * focusDistance * v
        )
        
        // Reset frame count
        constants.frameCount = 0
    }
    
    // MARK: Buffer Management
    
    func updateBuffers() {
        if !spheres.isEmpty {
            sphereBuffer = device.makeBuffer(bytes: spheres, length: MemoryLayout<Sphere>.stride * spheres.count, options: [])
        }
        
        if !materials.isEmpty {
            materialBuffer = device.makeBuffer(bytes: materials, length: MemoryLayout<Material>.stride * materials.count, options: [])
        }
        
        updateBVHBuffers()
        
        cameraBuffer = device.makeBuffer(bytes: &camera, length: MemoryLayout<Camera>.stride, options: [])
        constantsBuffer = device.makeBuffer(bytes: &constants, length: MemoryLayout<Constants>.stride, options: [])
    }
    
    func updateBVHBuffers() {
        if !bvhNodes.isEmpty {
            bvhNodeBuffer = device.makeBuffer(bytes: bvhNodes, length: MemoryLayout<BVHNode>.stride * bvhNodes.count, options: [])
        }
        
        if !bvhIndices.isEmpty {
            bvhIndexBuffer = device.makeBuffer(bytes: bvhIndices, length: MemoryLayout<Int32>.stride * bvhIndices.count, options: [])
        }
    }
    
    // MARK: MTKViewDelegate Methods
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        width = Int(size.width)
        height = Int(size.height)
        constants.width = UInt32(width)
        constants.height = UInt32(height)
        
        // Recreate texture with new size
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        resultTexture = device.makeTexture(descriptor: textureDescriptor)
        
        // Update camera for new aspect ratio
        setupCamera()
        updateBuffers()
    }
    
    func draw(in view: MTKView) {
        // Handle benchmarking
        if benchmarkMode {
            
            
            if frameCounter == 0 {
                benchmarkStartTime = CACurrentMediaTime()
                print("Benchmark \(benchmarkStep == 0 ? "without" : "with") BVH - starting...")
            }
            
            frameCounter += 1
            
            if frameCounter >= 100 {
                let endTime = CACurrentMediaTime()
                let elapsed = endTime - benchmarkStartTime
                let framesPerSecond = Double(frameCounter) / elapsed
                print("Benchmark \(benchmarkStep == 0 ? "without" : "with") BVH - completed:")
                print("  - \(frameCounter) frames in \(elapsed) seconds")
                print("  - \(framesPerSecond) FPS")
                
                // Reset for next benchmark or finish
                frameCounter = 0
                
                if benchmarkStep == 0 {
                    // Switch to BVH enabled for second benchmark
                    benchmarkStep = 1
                    useBVH = true
                    toggleBVH()
                } else {
                    // End benchmark mode
                    benchmarkMode = false
                    benchmarkStep = 0
                    print("Benchmark complete")
                }
            }
        }
        
        // Measure frame time
        let currentTime = CACurrentMediaTime()
        let frameDuration = currentTime - lastFrameTime
        lastFrameTime = currentTime
        let fps = 1.0 / frameDuration
        
        // Update FPS display
        if frameRateLabel == nil {
            frameRateLabel = NSTextField(frame: NSRect(x: 10, y: 10, width: 300, height: 20))
            frameRateLabel?.isEditable = false
            frameRateLabel?.isBordered = false
            frameRateLabel?.backgroundColor = NSColor.clear
            frameRateLabel?.textColor = NSColor.white
            frameRateLabel?.font = NSFont.systemFont(ofSize: 12, weight: .bold)
            view.addSubview(frameRateLabel!)
        }
        
        // Display performance metrics
        let samplesPerFrame = constants.width * constants.height * 4 // Assuming 4 samples per pixel
        let raysPerSecond = Double(samplesPerFrame) * fps / 1_000_000 // In millions
        frameRateLabel?.stringValue = String(format: "FPS: %.1f | Frame: %d | %.1f MRays/s | BVH: %@",
                                             fps, constants.frameCount, raysPerSecond, useBVH ? "ON" : "OFF")
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let drawable = view.currentDrawable else {
            return
        }
        
        // Add performance monitoring
        commandBuffer.addCompletedHandler { [weak self] buffer in
            let gpuTime = buffer.gpuEndTime - buffer.gpuStartTime
            
            if let frameCount = self?.constants.frameCount, frameCount % 60 == 0 {
                print("GPU execution time: \(gpuTime * 1000) ms | BVH: \(self?.useBVH ?? false ? "ON" : "OFF")")
            }
        }
        
        // Increment frame count
        constants.frameCount += 1
        memcpy(constantsBuffer.contents(), &constants, MemoryLayout<Constants>.stride)
        
        // Create compute encoder
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        
        computeEncoder.setComputePipelineState(computePipeline)
        
        // Set buffers
        computeEncoder.setBuffer(sphereBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(materialBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(cameraBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(constantsBuffer, offset: 0, index: 3)
        
        // Set BVH buffers if enabled
        if useBVH && !bvhNodes.isEmpty && !bvhIndices.isEmpty {
            computeEncoder.setBuffer(bvhNodeBuffer, offset: 0, index: 4)
            computeEncoder.setBuffer(bvhIndexBuffer, offset: 0, index: 5)
        }
        
        // Set textures
        computeEncoder.setTexture(resultTexture, index: 0)
        computeEncoder.setTexture(drawable.texture, index: 1)
        
        // Dispatch compute kernel
        let threadGroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadGroups = MTLSize(
            width: (width + threadGroupSize.width - 1) / threadGroupSize.width,
            height: (height + threadGroupSize.height - 1) / threadGroupSize.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
        computeEncoder.endEncoding()
        
        // Present and commit
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

// MARK: - SwiftUI Integration

struct MetalView: NSViewRepresentable {
    func makeNSView(context: Context) -> MTKView {
        let mtkView = MTKView()
        
        // Create ray tracer and set it up
        let rayTracer = MetalRayTracer()
        rayTracer.setup(view: mtkView)
        
        // Store ray tracer in coordinator
        context.coordinator.rayTracer = rayTracer
        
        return mtkView
    }
    
    func updateNSView(_ nsView: MTKView, context: Context) {
        // No updates needed
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var rayTracer: MetalRayTracer?
    }
}
