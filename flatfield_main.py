"""
Flatfield Correction Tool for Sony ARW RAW Images
Applies polynomial-based flatfield correction for RTI image sets in cultural heritage documentation

Corrects for combined lens vignetting + LED illumination patterns by applying
position-specific corrections derived from white board reference captures.

Author: Xavier Aure Calvet
Version: 1.1.0
License: GPL-3.0
"""

import os
import numpy as np
import cv2
import rawpy
import tifffile
import pickle
import argparse
import re
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures

__version__ = "1.1.0"

def parse_arguments():
    parser = argparse.ArgumentParser(
    description='Process Sony ARW raw images with flatfield correction and generate outputs.',
    epilog='Example: python flatfield_main.py --per-channel --coeffs-path ./flat_coeffs --parent-folder ./RTI_images'
    )
    parser.add_argument('--all', action='store_true', help='Generate all outputs (TIFF, JPG, and median)')
    parser.add_argument('--tiff', action='store_true', help='Generate TIFF outputs')
    parser.add_argument('--jpg', action='store_true', help='Generate JPG outputs')
    parser.add_argument('--median', action='store_true', help='Generate median image')
    parser.add_argument('--median-range', type=str, help='Range of images to use for median (e.g., "01-04" or "01-08")')
    parser.add_argument('--median-format', type=str, choices=['tiff', 'png', 'jpg'], default='tiff',
                        help='File format for median output (default: tiff)')
    parser.add_argument('--median-bit-depth', type=int, choices=[8, 16],
                        help='Override bit depth for median output (default: match --bit-depth)')
    parser.add_argument('--bit-depth', type=int, choices=[8, 16], default=16, help='Bit depth for output images (8 or 16, default: 16)')
    parser.add_argument('--num-cores', type=int, default=10, help='Number of CPU cores for parallel processing (default: 10)')
    parser.add_argument('--per-channel', action='store_true', 
                       help='Use per-channel RGB correction (must match coefficient generation mode)')
    parser.add_argument('--coeffs-path', type=str, required=True, help='Path to the folder containing coefficient files')
    parser.add_argument('--parent-folder', type=str, required=True, help='Path to the parent folder containing all image set folders (RTI_01, RTI_02, etc.)')
    args = parser.parse_args()
    
    # If no specific output is selected, default to all
    if not (args.all or args.tiff or args.jpg or args.median):
        args.all = True

    if args.median_bit_depth is None:
        args.median_bit_depth = args.bit_depth
        
    return args

def setup_directories(args, folder_name):
    """Setup input and output directories based on folder structure"""
    parent_folder = args.parent_folder
    
    # Setup paths
    img_dir = os.path.join(parent_folder, folder_name)
    coeffs_dir = args.coeffs_path
    
    # Determine bit depth suffix
    bit_suffix = f"_{args.bit_depth}bit"
    
    # Set up output directories
    output_dir_tiff = os.path.join(parent_folder, f"{folder_name}_flatfield_tiffs{bit_suffix}")
    jpg_dir_label = "jpgs" if args.bit_depth == 8 else "pngs"
    output_dir_jpg = os.path.join(parent_folder, f"{folder_name}_flatfield_{jpg_dir_label}{bit_suffix}")
    
    median_bit_suffix = f"_{args.median_bit_depth}bit"
    median_dir_label = {
        'tiff': 'tiffs',
        'png': 'pngs',
        'jpg': 'jpgs'
    }[args.median_format]
    median_extension = {
        'tiff': '.tif',
        'png': '.png',
        'jpg': '.jpg'
    }[args.median_format]

    # Median directory is shared across all folders
    median_output_dir = os.path.join(parent_folder, f"median_images_{median_dir_label}{median_bit_suffix}")
    output_median_file = f"{folder_name}{median_extension}"
    
    # Create dictionary of paths
    paths = {
        'img_dir': img_dir,
        'coeffs_dir': coeffs_dir,
        'output_dir_tiff': output_dir_tiff,
        'output_dir_jpg': output_dir_jpg,
        'median_output_dir': median_output_dir,
        'output_median_file': output_median_file
    }
    
    # Only create directories for the outputs that will be generated
    if args.all or args.tiff:
        os.makedirs(paths['output_dir_tiff'], exist_ok=True)
    if args.all or args.jpg:
        os.makedirs(paths['output_dir_jpg'], exist_ok=True)
    if args.all or args.median:
        os.makedirs(paths['median_output_dir'], exist_ok=True)
    
    return paths

# ---------------------------
# Fixed Parameters
# ---------------------------
# Optional cropping to match other processing workflows (e.g., RawTherapee exports)
# Set to camera native resolution (e.g., 9504×6336 for Sony A7R IV) to disable cropping
target_width = 7956
target_height = 5312
shift_x = 0
shift_y = 0
center_crop = True

# Supported RAW extensions (currently Sony only)
RAW_EXTENSIONS = ('.ARW', '.arw')

COLORSPACE_ENUM = getattr(rawpy, "ColorSpace", None)
LINEAR_OUTPUT_COLORSPACE = None
if COLORSPACE_ENUM is not None:
    LINEAR_OUTPUT_COLORSPACE = getattr(COLORSPACE_ENUM, "raw", None)
    if LINEAR_OUTPUT_COLORSPACE is None:
        LINEAR_OUTPUT_COLORSPACE = getattr(COLORSPACE_ENUM, "sRGB", None)
COORDINATE_NORMALIZATION_DEFAULT = '[-1, 1]'
COORDINATE_NORMALIZATION_PIXEL = 'pixel'


def _generate_2d_powers(degree: int) -> np.ndarray:
    """Replicate scikit-learn's PolynomialFeatures power ordering for 2D inputs."""
    powers = []
    for total_degree in range(degree + 1):
        for i in range(total_degree, -1, -1):
            powers.append((i, total_degree - i))
    return np.asarray(powers, dtype=np.int16)


def _infer_degree_from_terms(num_terms: int) -> int:
    """Infer polynomial degree from number of 2D terms."""
    for degree in range(0, 16):
        if (degree + 1) * (degree + 2) // 2 == num_terms:
            return degree
    raise ValueError(f"Unable to infer polynomial degree for {num_terms} terms")


def _prepare_coordinate_grids(shape, normalization):
    """Return coordinate grids matching the normalization used during fitting."""
    h, w = shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    if normalization == COORDINATE_NORMALIZATION_DEFAULT:
        denom_x = max(w - 1, 1)
        denom_y = max(h - 1, 1)
        xx = (xx / denom_x) * 2 - 1
        yy = (yy / denom_y) * 2 - 1
    return xx, yy


def _evaluate_poly_2d(coef, intercept, powers, coord_grids):
    """Evaluate a fitted polynomial surface without materialising a design matrix."""
    x, y = coord_grids
    coef = np.asarray(coef, dtype=np.float32).ravel()
    powers = np.asarray(powers, dtype=np.int16)

    max_i = int(powers[:, 0].max()) if powers.size else 0
    max_j = int(powers[:, 1].max()) if powers.size else 0

    x_pows = [np.ones_like(x, dtype=np.float32)]
    for _ in range(1, max_i + 1):
        x_pows.append(x_pows[-1] * x)

    y_pows = [np.ones_like(y, dtype=np.float32)]
    for _ in range(1, max_j + 1):
        y_pows.append(y_pows[-1] * y)

    surface = np.full_like(x, fill_value=float(intercept), dtype=np.float32)
    for c, (i, j) in zip(coef, powers):
        surface += c * x_pows[int(i)] * y_pows[int(j)]
    return surface

def flat_field_correction(image, fitted_flat):
    """
    Apply flatfield correction by dividing image by normalized flatfield pattern.
    
    Corrects for combined lens vignetting + LED illumination falloff captured
    in white board reference images.
    
    Args:
        image: Input image to correct
        fitted_flat: Fitted polynomial surface (brightness pattern from white board)
    
    Returns:
        Corrected image with uniform illumination
    """
    normalized_flat = fitted_flat / np.mean(fitted_flat)
    corr_image = cv2.divide(image.astype(np.float32), normalized_flat + 1e-5)
    corr_image = np.clip(corr_image, 0, 65535).astype(image.dtype)
    return corr_image

def save_jpeg_image(image, output_path):
    """Save image as standard 8-bit JPEG."""
    scaled_image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
    if not output_path.lower().endswith('.jpg'):
        output_path = os.path.splitext(output_path)[0] + '.jpg'
    cv2.imwrite(output_path, cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    return output_path


def save_png_image(image, output_path, bit_depth=16):
    """Save image as PNG at the requested bit depth (8 or 16)."""
    if not output_path.lower().endswith('.png'):
        output_path = os.path.splitext(output_path)[0] + '.png'

    if bit_depth == 8:
        scaled_image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))
        png_ready = cv2.cvtColor(scaled_image, cv2.COLOR_RGB2BGR)
    else:
        png_ready = cv2.cvtColor(image.astype(np.uint16), cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, png_ready, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return output_path

def save_tiff_image(image, output_path, bit_depth=16):
    """Save image as TIFF with specified bit depth"""
    if bit_depth == 8:
        # Convert to 8-bit
        scaled_image = cv2.convertScaleAbs(image, alpha=(255.0/65535.0))
        tifffile.imwrite(output_path, scaled_image)
    else:
        # Save as 16-bit
        tifffile.imwrite(output_path, image)

def load_polynomial_and_apply(image, coeffs_path, per_channel=False):
    """Load polynomial coefficients and apply flatfield correction without huge matrices."""

    def _warn_shape_mismatch(expected_shape, actual_shape, label):
        if expected_shape and tuple(expected_shape) != tuple(actual_shape):
            print(f"  WARNING: {label} shape mismatch. Fitted on {expected_shape}, applying to {actual_shape}.")

    with open(coeffs_path, 'rb') as f:
        coeffs_data = pickle.load(f)

    if isinstance(coeffs_data, dict) and 'per_channel' in coeffs_data:
        stored_per_channel = coeffs_data['per_channel']
        degree = coeffs_data.get('degree')
        stored_powers = coeffs_data.get('powers')
        if stored_powers is None:
            if degree is None:
                example = coeffs_data['channels'][0] if stored_per_channel else coeffs_data
                degree = _infer_degree_from_terms(len(example['coef']))
            stored_powers = _generate_2d_powers(degree)

        powers = np.asarray(stored_powers, dtype=np.int16)
        coord_norm = coeffs_data.get('coordinate_normalization', COORDINATE_NORMALIZATION_DEFAULT)

        if stored_per_channel != per_channel:
            print("WARNING: Coefficient mode mismatch!")
            print(f"  Coefficients are: {'per-channel' if stored_per_channel else 'grayscale'}")
            print(f"  Script expects: {'per-channel' if per_channel else 'grayscale'}")
            print("  Using coefficient mode from file to ensure compatibility.")
            per_channel = stored_per_channel

        if per_channel:
            if image.ndim == 3:
                coords = _prepare_coordinate_grids(image.shape[:2], coord_norm)
                corrected = np.zeros_like(image, dtype=np.float32)
                for channel_idx, channel_info in enumerate(coeffs_data['channels']):
                    channel_image = image[:, :, channel_idx]
                    _warn_shape_mismatch(channel_info.get('shape'), channel_image.shape, f"Channel {channel_info.get('channel', channel_idx)}")
                    fitted_flat = _evaluate_poly_2d(channel_info['coef'], channel_info['intercept'], powers, coords)
                    corrected[:, :, channel_idx] = flat_field_correction(channel_image, fitted_flat)
                return corrected

            # Single channel fallback with per-channel coefficients
            channel_info = coeffs_data['channels'][0]
            coords = _prepare_coordinate_grids(image.shape[:2], coord_norm)
            _warn_shape_mismatch(channel_info.get('shape'), image.shape, "Channel 0")
            fitted_flat = _evaluate_poly_2d(channel_info['coef'], channel_info['intercept'], powers, coords)
            return flat_field_correction(image, fitted_flat)

        # Grayscale coefficients
        coef = coeffs_data['coef']
        intercept = coeffs_data['intercept']
        coords = _prepare_coordinate_grids(image.shape[:2], coord_norm)
        _warn_shape_mismatch(coeffs_data.get('shape'), image.shape, "Grayscale")
        fitted_flat = _evaluate_poly_2d(coef, intercept, powers, coords)
        return flat_field_correction(image, fitted_flat)

    # Old tuple format fallback
    coef, intercept, shape = coeffs_data
    coef = np.asarray(coef, dtype=np.float32)
    degree = _infer_degree_from_terms(len(coef))
    powers = _generate_2d_powers(degree)
    coords = _prepare_coordinate_grids(image.shape[:2], COORDINATE_NORMALIZATION_PIXEL)
    _warn_shape_mismatch(shape, image.shape, "Legacy grayscale")
    fitted_flat = _evaluate_poly_2d(coef, intercept, powers, coords)
    return flat_field_correction(image, fitted_flat)

def crop_image(image, target_width, target_height, start_x=0, start_y=0, center_crop=False, shift_x=0, shift_y=0):
    """Crop image to specified dimensions with optional center cropping and shift"""
    if center_crop:
        center_x = image.shape[1] // 2
        center_y = image.shape[0] // 2
        start_x = max(center_x - target_width // 2 + shift_x, 0)
        start_y = max(center_y - target_height // 2 + shift_y, 0)
    else:
        start_x = max(start_x + shift_x, 0)
        start_y = max(start_y + shift_y, 0)
    end_x = start_x + target_width
    end_y = start_y + target_height
    
    # Make sure we don't go out of bounds
    if end_x > image.shape[1]:
        end_x = image.shape[1]
        start_x = max(0, end_x - target_width)
    if end_y > image.shape[0]:
        end_y = image.shape[0]
        start_y = max(0, end_y - target_height)
        
    return image[start_y:end_y, start_x:end_x]

def extract_image_number(filename):
    """Extract the numerical part from a filename."""
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    return None

def process_all_images(img_files, paths, median_range=None, per_channel=False):
    """
    Process all necessary raw images with flatfield correction
    Returns a dictionary mapping image filenames to corrected images
    """
    processed_images = {}
    
    # Keep original files for processing
    files_to_process = sorted(img_files)
    
    # Filter files by range if specified for median only
    median_files = files_to_process
    if median_range:
        try:
            # Parse the range (e.g., "01-04")
            start_num, end_num = median_range.split('-')
            
            # Filter files based on their numerical part
            filtered_files = []
            for file in img_files:
                num = extract_image_number(file)
                if num and start_num <= num <= end_num:
                    filtered_files.append(file)
            
            if not filtered_files:
                print(f"No files found in range {median_range} for median. Using all files.")
            else:
                print(f"Using {len(filtered_files)} files in range {median_range} for median")
                median_files = filtered_files  # Only update median_files, not all files
        except ValueError:
            print(f"Invalid range format: {median_range}. Using all files for median.")
    
    # Save both full and filtered lists in the returned dictionary
    print(f"Processing {len(files_to_process)} images with flatfield correction...")
    print(f"Correction mode: {'Per-channel (R, G, B separate)' if per_channel else 'Grayscale (single correction)'}")
    
    # Get num_cores from global scope (set in main)
    global num_cores
    
    # Process each raw file with flatfield correction
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_file = {executor.submit(process_single_raw_image, img_file, paths, per_channel): img_file for img_file in files_to_process}
        
        for future in concurrent.futures.as_completed(future_to_file):
            img_file = future_to_file[future]
            try:
                result = future.result()
                if result is not None:
                    processed_images[img_file] = result
            except Exception as e:
                print(f"Error processing image {img_file}: {e}")
    
    print(f"Successfully processed {len(processed_images)} images")
    return processed_images, median_files

def save_processed_images(processed_images, paths, generate_tiff=True, generate_jpg=True, bit_depth=16):
    """Save processed images as TIFF and/or JPG based on options"""
    if not (generate_tiff or generate_jpg):
        return
        
    for img_file, image in processed_images.items():
        # Save as TIFF if requested
        if generate_tiff:
            tiff_output_path = os.path.join(paths['output_dir_tiff'], os.path.basename(img_file)[:-4] + '.tif')
            save_tiff_image(image, tiff_output_path, bit_depth)
            print(f"TIFF file saved: {tiff_output_path}")
            
        # Save as JPG if requested
        if generate_jpg:
            if bit_depth == 8:
                base_output_path = os.path.join(paths['output_dir_jpg'], os.path.basename(img_file)[:-4] + '.jpg')
                saved_path = save_jpeg_image(image, base_output_path)
                print(f"JPG file saved: {saved_path}")
            else:
                base_output_path = os.path.join(paths['output_dir_jpg'], os.path.basename(img_file)[:-4] + '.png')
                saved_path = save_png_image(image, base_output_path, bit_depth=bit_depth)
                print(f"PNG file saved: {saved_path}")

def compute_and_save_median(filtered_processed_images, paths, median_format='tiff', bit_depth=16):
    """
    Compute and save median image from the filtered subset of processed images
    """
    if not filtered_processed_images:
        print("No images to process for median calculation.")
        return
        
    print(f"Computing median from {len(filtered_processed_images)} processed images...")
    images_list = list(filtered_processed_images.values())
    img_stack = np.stack(images_list, axis=0)
    median_image = np.median(img_stack, axis=0).astype(np.uint16)

    if median_format == 'jpg' and bit_depth != 8:
        print("WARNING: JPEG median output supports only 8-bit. Overriding requested bit depth to 8.")
        bit_depth = 8
    
    median_output_path = os.path.join(paths['median_output_dir'], paths['output_median_file'])

    if median_format == 'tiff':
        save_tiff_image(median_image, median_output_path, bit_depth)
        saved_path = median_output_path
    elif median_format == 'png':
        saved_path = save_png_image(median_image, median_output_path, bit_depth=bit_depth)
    else:
        saved_path = save_jpeg_image(median_image, median_output_path)

    print(f"Median image saved: {saved_path}")
    return median_image

def process_single_raw_image(img_file, paths, per_channel=False):
    """Process a single raw image and return the corrected image"""
    # Determine coefficient filename based on mode
    base_name = os.path.basename(img_file)[:-4]
    if per_channel:
        coeffs_path = os.path.join(paths['coeffs_dir'], base_name + '_coeffs_rgb.pkl')
        # Fallback to grayscale if per-channel not found
        if not os.path.exists(coeffs_path):
            coeffs_path = os.path.join(paths['coeffs_dir'], base_name + '_coeffs.pkl')
    else:
        coeffs_path = os.path.join(paths['coeffs_dir'], base_name + '_coeffs.pkl')
        # Fallback to per-channel if grayscale not found
        if not os.path.exists(coeffs_path):
            coeffs_path = os.path.join(paths['coeffs_dir'], base_name + '_coeffs_rgb.pkl')
    
    # Check if the coefficients file exists for the current image
    if os.path.exists(coeffs_path):
        print(f"Processing {img_file} with {coeffs_path}")
        
        # Load the image using rawpy
        with rawpy.imread(os.path.join(paths['img_dir'], img_file)) as raw:
            # Postprocess to convert the raw image to a visible spectrum image
            post_kwargs = dict(
                use_camera_wb=True,
                half_size=False,
                no_auto_bright=True,
                output_bps=16,
                gamma=(1, 1)
            )
            if LINEAR_OUTPUT_COLORSPACE is not None:
                post_kwargs['output_color'] = LINEAR_OUTPUT_COLORSPACE
            image = raw.postprocess(**post_kwargs)
            image_float = image.astype(np.float32)
            
            if per_channel:
                # Apply per-channel correction (R, G, B separately)
                corrected_image = load_polynomial_and_apply(image_float, coeffs_path, per_channel=True)
            else:
                # Apply grayscale correction to each channel
                for i in range(3):  # Assuming image is RGB
                    image_float[:, :, i] = load_polynomial_and_apply(image_float[:, :, i], coeffs_path, per_channel=False)
                corrected_image = image_float
            
            # After correction, clip the image values to ensure they are within valid range
            corrected_image = np.clip(corrected_image, 0, 65535).astype(np.uint16)
            
            # Crop the corrected image based on the specified parameters
            cropped_image = crop_image(corrected_image, target_width, target_height, center_crop=center_crop, shift_x=shift_x, shift_y=shift_y)
            
            return cropped_image
    else:
        # Log if the coefficients file does not exist for an image
        print(f"No coefficients found for {img_file}, skipping.")
        return None

def print_config(args, paths):
    """Print configuration summary"""
    print(f"Configuration:")
    print(f"Input directory: {paths['img_dir']}")
    print(f"Coefficients directory: {paths['coeffs_dir']}")
    print(f"Correction mode: {'Per-channel (R, G, B)' if args.per_channel else 'Grayscale'}")
    
    if args.all or args.tiff:
        print(f"Output TIFF directory: {paths['output_dir_tiff']}")
    if args.all or args.jpg:
        label = "JPG" if args.bit_depth == 8 else "PNG (requested via --jpg)"
        print(f"Output {label} directory: {paths['output_dir_jpg']}")
    
    print(f"Target width x height: {target_width} x {target_height}")
    print(f"Shift (X, Y): ({shift_x}, {shift_y})")
    print(f"Center crop: {center_crop}")
    
    if args.all or args.median:
        print(f"Median output directory: {paths['median_output_dir']}")
        print(f"Median output file: {paths['output_median_file']}")
        print(f"Median output format: {args.median_format.upper()} ({args.median_bit_depth}-bit)")
        if args.median_range:
            print(f"Median image range: {args.median_range}")
    
    print(f"Bit depth: {args.bit_depth}-bit")
    print(f"Number of cores: {num_cores}")
    print(f"Selected outputs: " + 
          ("All" if args.all else ", ".join(
              filter(None, [
                  "TIFF" if args.tiff else None,
                  "JPG" if args.jpg else None,
                  "Median" if args.median else None
              ])
          )))

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    
    # Set global num_cores from arguments
    num_cores = args.num_cores
    
    # Determine what to generate
    generate_tiff = args.all or args.tiff
    generate_jpg = args.all or args.jpg
    generate_median = args.all or args.median
    
    if args.median_format == 'jpg' and args.median_bit_depth != 8:
        print("WARNING: Median JPEG output supports only 8-bit. Overriding median bit depth to 8.")
        args.median_bit_depth = 8

    # Get all folders from the parent directory
    parent_folder = args.parent_folder
    if not os.path.isdir(parent_folder):
        print(f"ERROR: Parent folder '{parent_folder}' does not exist or is not a directory.")
        exit(1)
    
    # Find all subfolders with raw images (look for folders with .ARW files)
    image_folders = []
    for folder_name in sorted(os.listdir(parent_folder)):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            # Check if the folder contains raw images
            if os.path.exists(folder_path):
                raw_files = sorted(f for f in os.listdir(folder_path) if f.endswith(RAW_EXTENSIONS))
                if raw_files:
                    image_folders.append(folder_name)
    
    if not image_folders:
        print(f"ERROR: No folders with Sony ARW images found in '{parent_folder}'.")
        print(f"This tool is designed for Sony cameras. Please check your file paths.")
        exit(1)
    
    print("=" * 80)
    print(f"Flatfield Correction Tool v{__version__}")
    print("=" * 80)
    print(f"Found {len(image_folders)} image set folders: {', '.join(image_folders)}")
    print("=" * 80)
    
    # Process each folder
    for folder_index, folder_name in enumerate(image_folders):
        print(f"\nProcessing folder {folder_index+1}/{len(image_folders)}: {folder_name}")
        
        # Setup directories for this folder
        paths = setup_directories(args, folder_name)
        
        # Print configuration for this folder
        print_config(args, paths)
        
        # Get all raw files in this folder
        raw_files = sorted(f for f in os.listdir(paths['img_dir']) if f.endswith(RAW_EXTENSIONS))
        if not raw_files:
            print(f"Warning: No Sony ARW files found in {folder_name}, skipping.")
            print(f"This tool is designed for Sony cameras.")
            continue
        
        print(f"Found {len(raw_files)} raw images to process.")
        
        # Process all necessary images first
        processed_images, median_files = process_all_images(raw_files, paths, args.median_range, args.per_channel)
        
        # Generate requested outputs from processed images
        if generate_tiff or generate_jpg:
            save_processed_images(processed_images, paths, generate_tiff, generate_jpg, args.bit_depth)
        
        if generate_median:
            compute_and_save_median(
                {file: processed_images[file] for file in median_files if file in processed_images},
                paths,
                args.median_format,
                args.median_bit_depth
            )
            
    print("\n" + "=" * 80)
    print("All folders processed successfully.")
    print("=" * 80)
