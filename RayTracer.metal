#include <metal_stdlib>
using namespace metal;

// Define function constant for BVH toggle
constant bool use_bvh [[function_constant(0)]];

struct Sphere {
    float3 center;
    float radius;
    int material;
};

struct Ray {
    float3 origin;
    float3 direction;
};

struct Camera {
    float3 position;
    float3 lowerLeftCorner;
    float3 horizontal;
    float3 vertical;
};

struct Material {
    float3 albedo;
    int type;
    float fuzziness;
    float refractiveIndex;
};

struct HitRecord {
    float t;
    float3 p;
    float3 normal;
    int material;
};

struct Constants {
    uint width;
    uint height;
    uint sphereCount;
    uint materialCount;
    uint frameCount;
};

// BVH node structure for GPU
struct BVHNode {
    float3 min;              // Bounding box minimum
    int leftFirst;           // Index of left child or first primitive
    float3 max;              // Bounding box maximum
    int primitiveCount;      // Number of primitives (0 for interior nodes)
};

// Random number generation
float random(thread uint& seed) {
    seed = seed * 747796405 + 2891336453;
    uint result = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967295.0;
}

float3 random_unit_vector(thread uint& seed) {
    float z = random(seed) * 2.0 - 1.0;
    float a = random(seed) * 2.0 * 3.14159265;
    float r = sqrt(1.0 - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return float3(x, y, z);
}

// Ray-sphere intersection
bool hit_sphere(Sphere sphere, Ray ray, float t_min, float t_max, thread HitRecord& rec) {
    float3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float half_b = dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0) return false;
    
    float sqrtd = sqrt(discriminant);
    float root = (-half_b - sqrtd) / a;
    
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max)
            return false;
    }
    
    rec.t = root;
    rec.p = ray.origin + rec.t * ray.direction;
    rec.normal = (rec.p - sphere.center) / sphere.radius;
    rec.material = sphere.material;
    
    return true;
}

// Regular scene intersection (brute force)
bool hit_world(const device Sphere* spheres, uint sphere_count, Ray ray, float t_min, float t_max, thread HitRecord& rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    for (uint i = 0; i < sphere_count; i++) {
        if (hit_sphere(spheres[i], ray, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }
    
    return hit_anything;
}

// Bounding box intersection
bool hit_aabb(float3 aabb_min, float3 aabb_max, Ray ray, float t_min, float t_max) {
    float3 invD = 1.0f / ray.direction;
    float3 t0s = (aabb_min - ray.origin) * invD;
    float3 t1s = (aabb_max - ray.origin) * invD;
    float3 tsmaller = min(t0s, t1s);
    float3 tbigger = max(t0s, t1s);
    float tmin = max(t_min, max(tsmaller.x, max(tsmaller.y, tsmaller.z)));
    float tmax = min(t_max, min(tbigger.x, min(tbigger.y, tbigger.z)));
    return tmin < tmax;
}

// BVH traversal
bool hit_world_bvh(const device Sphere* spheres,
                  const device BVHNode* nodes,
                  const device int* primitive_indices,
                  Ray ray, float t_min, float t_max,
                  thread HitRecord& rec) {
    // Stack for non-recursive traversal
    int stack[64];
    int stack_ptr = 0;
    
    // Start with the root node
    stack[stack_ptr++] = 0;
    
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    // Traverse the BVH
    while (stack_ptr > 0) {
        int node_idx = stack[--stack_ptr];
        BVHNode node = nodes[node_idx];
        
        // Skip node if ray doesn't intersect bounding box
        if (!hit_aabb(node.min, node.max, ray, t_min, closest_so_far)) {
            continue;
        }
        
        if (node.primitiveCount > 0) {
            // Leaf node - check primitives
            for (int i = 0; i < node.primitiveCount; i++) {
                int sphere_idx = primitive_indices[node.leftFirst + i];
                if (hit_sphere(spheres[sphere_idx], ray, t_min, closest_so_far, rec)) {
                    hit_anything = true;
                    closest_so_far = rec.t;
                }
            }
        } else {
            // Interior node - push children
            stack[stack_ptr++] = node.leftFirst + 1; // Right child
            stack[stack_ptr++] = node.leftFirst;     // Left child
        }
    }
    
    return hit_anything;
}

// Main ray color calculation
float3 ray_color(Ray ray,
                 const device Sphere* spheres,
                 uint sphere_count,
                 const device Material* materials,
                 const device BVHNode* nodes,
                 const device int* primitive_indices,
                 thread uint& seed) {
    Ray current_ray = ray;
    float3 current_attenuation = float3(1.0, 1.0, 1.0);
    
    for (int i = 0; i < 5; i++) {  // Limit bounces for performance
        HitRecord rec;
        
        // Use either BVH or brute force ray casting
        bool hit = false;
        if (use_bvh && nodes != nullptr && primitive_indices != nullptr) {
            hit = hit_world_bvh(spheres, nodes, primitive_indices, current_ray, 0.001f, INFINITY, rec);
        } else {
            hit = hit_world(spheres, sphere_count, current_ray, 0.001f, INFINITY, rec);
        }
        
        if (hit) {
            // Apply material effects
            float3 target;
            float3 reflected;
            float3 refracted;
            
            int mat_type = materials[rec.material].type;
            float3 mat_color = materials[rec.material].albedo;
            float mat_fuzz = materials[rec.material].fuzziness;
            float mat_ref_idx = materials[rec.material].refractiveIndex;
            
            if (mat_type == 0) {  // Diffuse
                target = rec.p + rec.normal + random_unit_vector(seed);
                current_ray.origin = rec.p;
                current_ray.direction = normalize(target - rec.p);
                current_attenuation *= mat_color;
            }
            else if (mat_type == 1) {  // Metal
                reflected = reflect(normalize(current_ray.direction), rec.normal);
                reflected = normalize(reflected + mat_fuzz * random_unit_vector(seed));
                current_ray.origin = rec.p;
                current_ray.direction = reflected;
                current_attenuation *= mat_color;
                
                if (dot(current_ray.direction, rec.normal) <= 0) {
                    return float3(0, 0, 0);
                }
            }
            else if (mat_type == 2) {  // Glass
                float3 outward_normal;
                float ni_over_nt;
                float cosine;
                
                if (dot(current_ray.direction, rec.normal) > 0) {
                    outward_normal = -rec.normal;
                    ni_over_nt = mat_ref_idx;
                    cosine = mat_ref_idx * dot(current_ray.direction, rec.normal) / length(current_ray.direction);
                } else {
                    outward_normal = rec.normal;
                    ni_over_nt = 1.0 / mat_ref_idx;
                    cosine = -dot(current_ray.direction, rec.normal) / length(current_ray.direction);
                }
                
                float reflect_prob = 1.0;
                float3 refracted;
                
                // Try to refract
                float3 uv = normalize(current_ray.direction);
                float dt = dot(uv, outward_normal);
                float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1.0 - dt * dt);
                
                if (discriminant > 0) {
                    refracted = ni_over_nt * (uv - outward_normal * dt) - outward_normal * sqrt(discriminant);
                    
                    // Schlick approximation for reflectance
                    float r0 = (1 - ni_over_nt) / (1 + ni_over_nt);
                    r0 = r0 * r0;
                    reflect_prob = r0 + (1 - r0) * pow((1 - cosine), 5);
                }
                
                if (random(seed) < reflect_prob) {
                    reflected = reflect(normalize(current_ray.direction), rec.normal);
                    current_ray.origin = rec.p;
                    current_ray.direction = reflected;
                } else {
                    current_ray.origin = rec.p;
                    current_ray.direction = normalize(refracted);
                }
            }
        }
        else {
            // Sky background
            float3 unit_direction = normalize(current_ray.direction);
            float t = 0.5 * (unit_direction.y + 1.0);
            float3 sky_color = (1.0 - t) * float3(1.0) + t * float3(0.5, 0.7, 1.0);
            return current_attenuation * sky_color;
        }
    }
    
    return float3(0);  // Too many bounces, return black
}

// Main ray tracing kernel
kernel void rayTracing(
    const device Sphere* spheres [[buffer(0)]],
    const device Material* materials [[buffer(1)]],
    const device Camera* camera [[buffer(2)]],
    const device Constants& constants [[buffer(3)]],
    const device BVHNode* bvh_nodes [[buffer(4), function_constant(use_bvh)]],
    const device int* bvh_indices [[buffer(5), function_constant(use_bvh)]],
    texture2d<float, access::read_write> resultTexture [[texture(0)]],
    texture2d<float, access::write> finalTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint width = constants.width;
    uint height = constants.height;
    uint sphere_count = constants.sphereCount;
    uint frame_count = constants.frameCount;
    
    // Check bounds
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    // Initialize seed for random number generator
    uint seed = (gid.y * width + gid.x) * frame_count + frame_count;
    
    // Debug pattern for the corner
    float3 debug_color = float3(float(gid.x) / width, float(gid.y) / height, 0.5);
    
    // Sample multiple rays for anti-aliasing
    float3 pixel_color = float3(0.0);
    const int samples = 4;  // Reduced for performance
    
    for (int s = 0; s < samples; s++) {
        float u = float(gid.x + random(seed)) / float(width);
        float v = float(height - gid.y - 1 + random(seed)) / float(height);
        
        Ray ray;
        ray.origin = camera->position;
        ray.direction = normalize(camera->lowerLeftCorner + u * camera->horizontal + v * camera->vertical - camera->position);
        
        // Call the ray_color function with correct arguments
        pixel_color += ray_color(ray, spheres, sphere_count, materials,
                                use_bvh ? bvh_nodes : nullptr,
                                use_bvh ? bvh_indices : nullptr,
                                seed);
    }
    
    // Divide by number of samples and gamma-correct
    pixel_color = pixel_color / float(samples);
    pixel_color = float3(sqrt(pixel_color.x), sqrt(pixel_color.y), sqrt(pixel_color.z));
    
    // Show debug pattern in corner
    if (gid.x < 100 && gid.y < 100) {
        pixel_color = debug_color;
    }
    
    // Progressive rendering
    if (frame_count > 1) {
        float3 prev_color = resultTexture.read(gid).xyz;
        float blend_factor = 1.0 / float(frame_count);
        pixel_color = prev_color * (1.0 - blend_factor) + pixel_color * blend_factor;
    }
    
    // Write to result texture for accumulation
    resultTexture.write(float4(pixel_color, 1.0), gid);
    
    // Write to final texture for display
    finalTexture.write(float4(pixel_color, 1.0), gid);
}
