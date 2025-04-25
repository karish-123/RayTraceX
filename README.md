# RayTraceX

**RayTraceX** is a hybrid ray tracer built with **Swift** and **Metal**. It performs **BVH-based acceleration on the CPU** and **ray traversal and shading on the GPU**, showcasing a heterogeneous rendering pipeline optimized for performance and clarity.

---

## Features

- Heterogeneous architecture (CPU + GPU)
- BVH acceleration structure on CPU
- Fast ray traversal and shading via Metal
- Material support: diffuse, metal, dielectric
- Progressive rendering with anti-aliasing
- Built in Swift and Metal Shading Language (MSL)

---

## Architecture Overview

- Scene setup and camera management in Swift
- BVH construction happens on CPU and is passed to the GPU
- Ray traversal and material shading executed in parallel on GPU
- Final image buffer is rendered and displayed using Metal

---

## Requirements

- macOS with Metal support
- Xcode 14 or later
- M1/M2/M3/M4 Mac recommended for GPU acceleration

