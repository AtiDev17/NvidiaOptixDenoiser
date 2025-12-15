import sys
import argparse
import optix
import cupy as cp
import numpy as np
from PIL import Image
import traceback
import time


# --- Color Space Helpers ---
def srgb_to_linear(img_data):
    """Converts sRGB (0.0-1.0) to Linear space."""
    return cp.power(img_data, 2.2)


def linear_to_srgb(img_data):
    """Converts Linear space to sRGB (0.0-1.0)."""
    img_data = cp.clip(img_data, 0.0, 1.0)
    return cp.power(img_data, 1.0 / 2.2)


def compute_intensity(img_data):
    """
    Computes the logarithmic average intensity of the image using CuPy.
    Matches OptiX behavior: exp( avg( log( luminance + 1e-6 ) ) )
    """
    # Subsample for speed and lower memory usage on huge images
    # Step of 8 means 1/64th of the pixels, which is statistically sufficient for intensity
    sub_data = img_data[::8, ::8, :]

    # Weights for luminance (Rec. 709 linear)
    if sub_data.shape[2] >= 3:
        lum = 0.2126 * sub_data[:, :, 0] + 0.7152 * sub_data[:, :, 1] + 0.0722 * sub_data[:, :, 2]
    else:
        lum = sub_data[:, :, 0]

    log_lum = cp.log(lum + 1e-6)
    avg_log_lum = cp.mean(log_lum)
    intensity = cp.exp(avg_log_lum)

    # Return as single-element float32 array
    return cp.array([intensity], dtype=cp.float32)


def _optix_log_callback(level, tag, message):
    print(f"[{level}][{tag}]: {message}")


class OptixImageWrapper:
    """Helper to manage CuPy array and OptixImage2D struct."""

    def __init__(self, cupy_array):
        self.array = cupy_array
        self.height, self.width, self.channels = cupy_array.shape

        self.image2d = optix.Image2D()
        self.image2d.data = cupy_array.data.ptr
        self.image2d.width = self.width
        self.image2d.height = self.height
        # 4 bytes per float component
        self.image2d.pixelStrideInBytes = self.channels * 4
        self.image2d.rowStrideInBytes = self.width * self.image2d.pixelStrideInBytes

        if self.channels == 4:
            self.image2d.format = optix.PIXEL_FORMAT_FLOAT4
        elif self.channels == 3:
            self.image2d.format = optix.PIXEL_FORMAT_FLOAT3
        else:
            raise ValueError(f"Unsupported channel count: {self.channels}. Must be 3 or 4.")

    def create_view(self, x, y, w, h):
        """Creates an OptixImage2D view into the buffer."""
        view = optix.Image2D()
        offset = (y * self.image2d.rowStrideInBytes) + (x * self.image2d.pixelStrideInBytes)
        view.data = self.image2d.data + offset
        view.width = w
        view.height = h
        view.pixelStrideInBytes = self.image2d.pixelStrideInBytes
        view.rowStrideInBytes = self.image2d.rowStrideInBytes
        view.format = self.image2d.format
        return view


def load_image(path, is_srgb=True):
    """
    Loads an image to GPU as float32.
    Converts to RGBA if needed.
    Linearizes color if is_srgb is True (keeps Alpha linear).
    """
    try:
        img = Image.open(path).convert('RGBA')
    except Exception as e:
        raise ValueError(f"Failed to load {path}: {e}")

    # Normalize to 0-1
    host_data = np.array(img).astype(np.float32) / 255.0
    gpu_data = cp.asarray(host_data)

    if is_srgb:
        # Convert RGB to linear, keep Alpha as-is
        gpu_data[:, :, :3] = srgb_to_linear(gpu_data[:, :, :3])

    return gpu_data


def get_optimal_tile_size(denoiser, available_mem):
    """
    Calculates the largest safe tile size (power of 2) that fits in available VRAM.
    Consider scratch size + overlap overhead.
    """
    candidates = [512, 1024, 2048, 4096, 8192]
    best_size = 512

    # Safety margin: Leave 20% of free memory untouched for driver overhead/fragmentation
    safe_mem = available_mem * 0.8

    for size in candidates:
        try:
            # Calculate requirements for this tile size
            # We must account for the overlap area which OptiX adds internally to the setup dimensions
            # Usually overlap is constant or function of model, check for 512x512 to be safe or just ask
            # Note: denoiser.computeMemoryResources needs width/height including overlap for setup?
            # Actually, computeMemoryResources(w, h) gives requirements for processing an image of size w*h
            # The denoiser setup requires (tile_width + 2*overlap, tile_height + 2*overlap)

            # First, get overlap for this target tile size
            res_check = denoiser.computeMemoryResources(size, size)
            overlap = res_check.overlapWindowSizeInPixels

            setup_w = size + 2 * overlap
            setup_h = size + 2 * overlap

            # Now get actual memory requirement for this setup
            reqs = denoiser.computeMemoryResources(setup_w, setup_h)

            total_needed = reqs.stateSizeInBytes + reqs.withOverlapScratchSizeInBytes

            if total_needed < safe_mem:
                best_size = size
            else:
                # If this size doesn't fit, larger ones won't either
                break

        except Exception:
            # If computeMemoryResources throws (e.g. too large for internal model), stop
            break

    return best_size


def denoise(args):
    denoiser = None
    context = None

    try:
        # 1. Init
        cp.cuda.Device(0).use()

        ctx_options = optix.DeviceContextOptions()
        ctx_options.logCallbackFunction = _optix_log_callback
        ctx_options.logCallbackLevel = 3  # Info
        context = optix.deviceContextCreate(0, ctx_options)

        # 2. Load Input
        print(f"Loading input: {args.input}")
        input_data = load_image(args.input, is_srgb=True)
        input_wrapper = OptixImageWrapper(input_data)

        width = input_wrapper.width
        height = input_wrapper.height
        print(f"Resolution: {width}x{height} ({width * height / 1e6:.2f} Megapixels)")

        # Prepare Output
        output_data = cp.empty_like(input_data)
        output_wrapper = OptixImageWrapper(output_data)

        # 3. Load Guides (Optional)
        albedo_wrapper = None
        if args.albedo:
            print(f"Loading Albedo Guide: {args.albedo}")
            albedo_data = load_image(args.albedo, is_srgb=True)  # Albedo is color
            albedo_wrapper = OptixImageWrapper(albedo_data)
            if albedo_wrapper.width != width or albedo_wrapper.height != height:
                raise ValueError("Albedo resolution mismatch")

        normal_wrapper = None
        if args.normal:
            print(f"Loading Normal Guide: {args.normal}")
            # Normals are data, usually linear.
            # Note: OptiX expects Camera-Space normals. Using Tangent-Space might give mixed results.
            normal_data = load_image(args.normal, is_srgb=False)
            normal_wrapper = OptixImageWrapper(normal_data)
            if normal_wrapper.width != width or normal_wrapper.height != height:
                raise ValueError("Normal resolution mismatch")

        # 4. Setup Denoiser
        options = optix.DenoiserOptions()
        options.guideAlbedo = 1 if albedo_wrapper else 0
        options.guideNormal = 1 if normal_wrapper else 0
        # Denoise the alpha channel as well (crucial for sprites/cutouts)
        options.denoiseAlpha = optix.DENOISER_ALPHA_MODE_DENOISE

        # Use HDR model for best quality on linearized data
        model_kind = optix.DENOISER_MODEL_KIND_HDR
        denoiser = context.denoiserCreate(model_kind, options)

        # 5. Compute Intensity (Critical for HDR model stability)
        print("Computing intensity...")
        # Use CuPy implementation to avoid OptiX memory resource limits on large images
        intensity_arr = compute_intensity(input_data)
        # We need to keep the array alive and use its pointer
        intensity_mem = intensity_arr.data

        # 6. Tiling Setup
        tile_w, tile_h = args.tile_size, args.tile_size

        if tile_w <= 0:
            print("calculating optimal tile size...")
            free_mem, total_mem = cp.cuda.Device(0).mem_info
            # Note: free_mem includes the space already taken by input/output images
            # So it truly reflects what is LEFT for the denoiser scratch buffers.
            tile_w = get_optimal_tile_size(denoiser, free_mem)
            tile_h = tile_w
            print(f"Optimal tile size calculated: {tile_w}x{tile_h}")

        # Get overlap requirement
        sizes = denoiser.computeMemoryResources(tile_w, tile_h)
        overlap = sizes.overlapWindowSizeInPixels

        # Allocation sizes for Tile + Overlap
        setup_w = tile_w + 2 * overlap
        setup_h = tile_h + 2 * overlap
        tile_sizes = denoiser.computeMemoryResources(setup_w, setup_h)

        print(f"Tile Size: {tile_w}x{tile_h} (Overlap: {overlap})")

        scratch_mem = cp.cuda.alloc(tile_sizes.withOverlapScratchSizeInBytes)
        state_mem = cp.cuda.alloc(tile_sizes.stateSizeInBytes)

        # Setup denoiser for the max tile size
        denoiser.setup(
            0,
            setup_w, setup_h,
            state_mem.ptr, tile_sizes.stateSizeInBytes,
            scratch_mem.ptr, tile_sizes.withOverlapScratchSizeInBytes
        )

        # 7. Execution
        params = optix.DenoiserParams()
        params.hdrIntensity = intensity_mem.ptr
        params.blendFactor = 0.0  # 0.0 = Fully Denoised

        print("Denoising (Manual Tiling)...")
        start_time = time.time()

        # Temp buffer for one tile (including overlap)
        max_w = tile_w + 2 * overlap
        max_h = tile_h + 2 * overlap
        tile_out_mem = cp.empty((max_h, max_w, input_wrapper.channels), dtype=cp.float32)
        tile_out_wrapper = OptixImageWrapper(tile_out_mem)

        total_tiles_x = (width + tile_w - 1) // tile_w
        total_tiles_y = (height + tile_h - 1) // tile_h
        total_tiles = total_tiles_x * total_tiles_y
        processed = 0

        try:
            for ty in range(total_tiles_y):
                for tx in range(total_tiles_x):
                    out_x, out_y = tx * tile_w, ty * tile_h
                    out_w, out_h = min(tile_w, width - out_x), min(tile_h, height - out_y)

                    win_x, win_y = max(0, out_x - overlap), max(0, out_y - overlap)
                    win_end_x, win_end_y = min(width, out_x + out_w + overlap), min(height, out_y + out_h + overlap)
                    win_w, win_h = win_end_x - win_x, win_end_y - win_y

                    # Create Views
                    guide_layer_tile = optix.DenoiserGuideLayer()
                    if albedo_wrapper: guide_layer_tile.albedo = albedo_wrapper.create_view(win_x, win_y, win_w, win_h)
                    if normal_wrapper: guide_layer_tile.normal = normal_wrapper.create_view(win_x, win_y, win_w, win_h)

                    layer_tile = optix.DenoiserLayer()
                    layer_tile.input = input_wrapper.create_view(win_x, win_y, win_w, win_h)
                    layer_tile.output = tile_out_wrapper.create_view(0, 0, win_w, win_h)

                    denoiser.invoke(
                        0, params, state_mem.ptr, tile_sizes.stateSizeInBytes,
                        guide_layer_tile, [layer_tile], 0, 0,
                        scratch_mem.ptr, tile_sizes.withOverlapScratchSizeInBytes
                    )

                    # Copy Valid Region
                    valid_off_x, valid_off_y = out_x - win_x, out_y - win_y
                    output_data[out_y: out_y + out_h, out_x: out_x + out_w] = \
                        tile_out_mem[valid_off_y: valid_off_y + out_h, valid_off_x: valid_off_x + out_w]

                    processed += 1
                    print(f"Processed tile {processed}/{total_tiles}...", end='\r')

            print("")  # Newline
        except KeyboardInterrupt:
            print("\nAborted by user.")
            return

        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        print(f"Denoising took: {end_time - start_time:.2f} seconds")

        # 8. Save Output
        cp.cuda.Stream.null.synchronize()
        print("Saving...")

        # Convert Linear -> sRGB
        res_srgb = linear_to_srgb(output_data)

        # Convert to uint8
        res_uint8 = cp.asnumpy(res_srgb * 255.0)
        res_uint8 = np.clip(res_uint8, 0, 255).astype(np.uint8)

        img = Image.fromarray(res_uint8)
        if args.output.lower().endswith(('.jpg', '.jpeg')):
            img = img.convert('RGB')

        img.save(args.output)
        print(f"Done. Saved to {args.output}")

    except Exception as e:
        print("\n[ERROR]")
        traceback.print_exc()

    finally:
        if denoiser: denoiser.destroy()
        if context: context.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OptiX AI Denoiser for 3D Renders")
    parser.add_argument("input", help="Input image (PNG/EXR/JPG)")
    parser.add_argument("output", help="Output image path")
    parser.add_argument("--albedo", help="Optional Albedo/Diffuse color guide image")
    parser.add_argument("--normal", help="Optional Normal guide image (Camera Space preferred)")
    parser.add_argument("--tile-size", type=int, default=0,
                        help="Tile size (0 = Auto, >0 = Manual). Auto calculates based on VRAM.")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    denoise(args)
