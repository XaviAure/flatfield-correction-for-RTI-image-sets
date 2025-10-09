"""
Polynomial Coefficient Generator for Flatfield Correction
Generates polynomial coefficients from Sony ARW white board reference images

For RTI in Cultural Heritage: Capture white board at object position for each LED to model 
combined lens vignetting + LED illumination patterns (hotspots, falloff, colour variations)

Supports polynomial degrees 3-9 (default 5) to match lighting complexity.

Author: Xavier Aure Calvet
Version: 1.1.0
"""

import os
import argparse
import numpy as np
import cv2
import rawpy
import pickle  # For saving polynomial coefficients
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

COLORSPACE_ENUM = getattr(rawpy, "ColorSpace", None)
LINEAR_OUTPUT_COLORSPACE = None
if COLORSPACE_ENUM is not None:
    LINEAR_OUTPUT_COLORSPACE = getattr(COLORSPACE_ENUM, "raw", None)
    if LINEAR_OUTPUT_COLORSPACE is None:
        LINEAR_OUTPUT_COLORSPACE = getattr(COLORSPACE_ENUM, "sRGB", None)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate polynomial coefficients from Sony ARW white board flatfield images.',
        epilog='Example: python calculate_polynomial_coefficients.py --per-channel'
    )
    parser.add_argument('--per-channel', action='store_true', 
                       help='Fit separate polynomials for R, G, B channels (more accurate for color-dependent effects)')
    parser.add_argument('--degree', type=int, default=5, choices=[3, 5, 7, 9],
                       help='Polynomial degree (default: 5). Use 7 or 9 for sharp LED hotspots')
    parser.add_argument('--flat-dir', type=str, default='RTI_flats',
                       help='Directory containing white board ARW images (default: RTI_flats)')
    parser.add_argument('--output-dir', type=str, default='flat_coeffs',
                       help='Output directory for coefficient files (default: flat_coeffs)')
    return parser.parse_args()

def fit_polynomial_to_flat_and_save(flat, flat_file_name, save_dir, degree=5, per_channel=False):
    """
    Fit a polynomial surface to a white board flatfield image and save coefficients.
    
    The polynomial models the combined optical + illumination pattern as:
    brightness_pattern = c₀ + c₁x + c₂y + c₃x² + c₄xy + c₅y² + ... 
    
    This captures:
    - Lens vignetting (optical artifacts)
    - LED illumination falloff (inverse square law)
    - LED hotspot patterns specific to each position
    - Color-dependent effects (if per_channel=True)
    
    Args:
        flat: Flatfield image (2D grayscale or 3D RGB numpy array)
        flat_file_name: Base filename for saving coefficients
        save_dir: Directory to save coefficient files
        degree: Polynomial degree (default: 5, producing 21 terms)
        per_channel: If True, fit R, G, B separately; if False, use grayscale
    """
    # Capture the polynomial exponent mapping once so evaluation can avoid rebuilding PolynomialFeatures
    dummy_poly = PolynomialFeatures(degree=degree)
    dummy_poly.fit(np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32))
    powers = dummy_poly.powers_.astype(np.int16, copy=False)

    if per_channel and len(flat.shape) == 3:
        # Fit separate polynomials for each RGB channel
        print(f"  Fitting separate polynomials for R, G, B channels (degree {degree})")
        coeffs_list = []
        
        for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
            channel_data = flat[:, :, channel_idx].astype(np.float32)
            h, w = channel_data.shape
            
            # Generate grid coordinates
            yy, xx = np.indices((h, w))
            denom_x = max(w - 1, 1)
            denom_y = max(h - 1, 1)
            x = (xx.astype(np.float32) / denom_x) * 2 - 1
            y = (yy.astype(np.float32) / denom_y) * 2 - 1
            features = np.stack([x.ravel(), y.ravel()], axis=1)
            
            # Generate polynomial features
            poly = PolynomialFeatures(degree=degree)
            poly_features = poly.fit_transform(features)
            poly_features = poly_features.astype(np.float32, copy=False)
            
            # Fit a linear model
            model = LinearRegression()
            model.fit(poly_features, channel_data.ravel())
            
            coeffs_list.append({
                'coef': model.coef_.astype(np.float32),
                'intercept': float(model.intercept_),
                'shape': channel_data.shape,
                'channel': channel_name
            })
        
        # Save all channel coefficients
        coeffs_path = os.path.join(save_dir, flat_file_name + '_coeffs_rgb.pkl')
        with open(coeffs_path, 'wb') as f:
            pickle.dump({
                'per_channel': True,
                'degree': degree,
                'channels': coeffs_list,
                'powers': powers,
                'coordinate_normalization': '[-1, 1]'
            }, f)
        
    else:
        # Convert to grayscale if RGB
        if len(flat.shape) == 3:
            flat = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
            print(f"  Fitting grayscale polynomial (degree {degree})")
        
        flat = flat.astype(np.float32)
        h, w = flat.shape
        # Generate grid coordinates
        yy, xx = np.indices((h, w))
        denom_x = max(w - 1, 1)
        denom_y = max(h - 1, 1)
        x = (xx.astype(np.float32) / denom_x) * 2 - 1
        y = (yy.astype(np.float32) / denom_y) * 2 - 1
        features = np.stack([x.ravel(), y.ravel()], axis=1)
        
        # Generate polynomial features
        poly = PolynomialFeatures(degree=degree)
        poly_features = poly.fit_transform(features)
        poly_features = poly_features.astype(np.float32, copy=False)
        
        # Fit a linear model
        model = LinearRegression()
        model.fit(poly_features, flat.ravel())
        
        # Save the model coefficients
        coeffs_path = os.path.join(save_dir, flat_file_name + '_coeffs.pkl')
        with open(coeffs_path, 'wb') as f:
            pickle.dump({
                'per_channel': False,
                'degree': degree,
                'coef': model.coef_.astype(np.float32),
                'intercept': float(model.intercept_),
                'shape': flat.shape,
                'powers': powers,
                'coordinate_normalization': '[-1, 1]'
            }, f)
    
    print(f"  Coefficients saved: {coeffs_path}")

def process_and_save_flat_files(flat_dir, save_dir, degree=5, per_channel=False):
    """
    Process all white board ARW files in flat_dir and generate polynomial coefficients.
    
    Each white board image should correspond to one LED position in your RTI setup.
    
    Args:
        flat_dir: Directory containing white board flatfield ARW images
        save_dir: Directory to save coefficient files
        degree: Polynomial degree (default: 5)
        per_channel: If True, fit R, G, B separately for color accuracy
    """
    # Find all ARW files
    flat_files = sorted(f for f in os.listdir(flat_dir) if f.upper().endswith('.ARW'))
    
    if not flat_files:
        print(f"ERROR: No Sony ARW files found in '{flat_dir}'")
        print("This tool is designed for Sony cameras. Please check your file paths.")
        return
    
    num_coeffs = int((degree+1)*(degree+2)/2)
    print(f"Found {len(flat_files)} white board images to process")
    print(f"Polynomial degree: {degree} (generates {num_coeffs} coefficients)")
    print(f"Processing mode: {'Per-channel (R, G, B separate)' if per_channel else 'Grayscale (single correction)'}")
    if per_channel:
        print(f"  → Total coefficients per image: {num_coeffs * 3} ({num_coeffs} × 3 channels)")
    print("-" * 60)
    
    for flat_file in flat_files:
        try:
            print(f"Processing: {flat_file}")
            with rawpy.imread(os.path.join(flat_dir, flat_file)) as raw:
                post_kwargs = dict(
                    use_camera_wb=True,
                    half_size=False,
                    no_auto_bright=True,
                    output_bps=16,
                    gamma=(1, 1)
                )
                if LINEAR_OUTPUT_COLORSPACE is not None:
                    post_kwargs['output_color'] = LINEAR_OUTPUT_COLORSPACE
                flat = raw.postprocess(**post_kwargs)
                fit_polynomial_to_flat_and_save(flat, flat_file[:-4], save_dir, degree=degree, per_channel=per_channel)
        except Exception as e:
            print(f"  ERROR processing {flat_file}: {e}")
            continue
    
    print("-" * 60)
    print(f"Processing complete. Coefficients saved in: {save_dir}")

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    
    flat_dir = args.flat_dir
    coeffs_save_dir = args.output_dir
    
    # Check if input directory exists
    if not os.path.exists(flat_dir):
        print(f"ERROR: Flatfield directory '{flat_dir}' does not exist!")
        print(f"Please create it and add your Sony ARW white board images.")
        exit(1)
    
    # Create output directory
    os.makedirs(coeffs_save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Polynomial Coefficient Generator for Flatfield Correction")
    print("=" * 60)
    print(f"Input directory:  {flat_dir}")
    print(f"Output directory: {coeffs_save_dir}")
    print("=" * 60)
    
    process_and_save_flat_files(flat_dir, coeffs_save_dir, degree=args.degree, per_channel=args.per_channel)
    
    print("\nNOTE: When processing images, use the matching mode:")
    if args.per_channel:
        print("  python flatfield_main.py --per-channel --coeffs-path ./flat_coeffs ...")
    else:
        print("  python flatfield_main.py --coeffs-path ./flat_coeffs ...")
