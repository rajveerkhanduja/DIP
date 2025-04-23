# DIP

# Digital Image Processing Lab Experiments

## Experiment 1: Basic Image Operations

### 1.a) Read, Display and Write Color Image in Other Formats

*Algorithm:*
1. Load the input color image using appropriate library functions
2. Display the loaded image on screen
3. Convert the image to different formats (JPG, PNG, BMP, TIFF)
4. Save the converted images to disk with appropriate extensions

*Concept:*
- Digital images are stored as matrices of pixel values
- Different image formats use different compression techniques:
  - JPG: Lossy compression suitable for photographs
  - PNG: Lossless compression with transparency support
  - BMP: Uncompressed format that preserves all details
  - TIFF: Flexible format supporting multiple images and layers

### 1.b) Find RED, GREEN and BLUE Planes of Color Image

*Algorithm:*
1. Load the input color image
2. Create three separate matrices of the same dimensions as the original image
3. For each pixel in the image:
   - Copy the red channel value to the first matrix, set other channels to zero
   - Copy the green channel value to the second matrix, set other channels to zero
   - Copy the blue channel value to the third matrix, set other channels to zero
4. Display the three resulting images (red, green, and blue planes)

*Concept:*
- Color images use the RGB color model where each pixel has three values
- Each plane shows the intensity distribution of that particular color
- The combination of all three planes produces the full-color image
- Areas appearing bright in a specific plane indicate high intensity of that color

### 1.c) Convert Color Image to Grayscale and Binary Image

*Algorithm for Grayscale Conversion:*
1. Load the input color image
2. Create a new matrix of the same dimensions
3. For each pixel, calculate grayscale value using the formula:
   Gray = 0.299×Red + 0.587×Green + 0.114×Blue
4. Assign this value to the corresponding pixel in the new matrix
5. Display the resulting grayscale image

*Algorithm for Binary Conversion:*
1. Convert the color image to grayscale using the above algorithm
2. Determine a threshold value (can be fixed, e.g., 128, or calculated using methods like Otsu's)
3. Create a new binary matrix of the same dimensions
4. For each pixel in the grayscale image:
   - If grayscale value > threshold, set binary pixel to 1 (white)
   - Otherwise, set binary pixel to 0 (black)
5. Display the resulting binary image

*Concept:*
- Grayscale images have only intensity information but no color
- The RGB-to-grayscale conversion weights reflect human perception of brightness
- Binary images contain only two values (0 and 1) representing black and white
- Thresholding is the simplest segmentation method for creating binary images

### 1.d) Resize Image by One Half and One Quarter

*Algorithm:*
1. Load the input image
2. Create two new matrices with dimensions:
   - Half-size matrix: width/2 × height/2
   - Quarter-size matrix: width/4 × height/4
3. For each pixel position (i,j) in the new matrices:
   - Map to position (2i,2j) in original for half-size
   - Map to position (4i,4j) in original for quarter-size
   - Alternative: Use interpolation methods for better quality
4. Display the original, half-size, and quarter-size images

*Concept:*
- Image resizing changes the number of pixels in the image
- Downsampling (reducing size) can lead to loss of information
- Simple methods like nearest neighbor lose detail
- More sophisticated methods like bilinear or bicubic interpolation preserve more details
- Resizing affects storage size and processing requirements

### 1.e) Image Rotation by 45, 90, and 180 Degrees

*Algorithm:*
1. Load the input image with dimensions width×height
2. For 90° rotation:
   - Create a new matrix with dimensions height×width
   - For each pixel (i,j) in original, place at (height-j-1,i) in new matrix
3. For 180° rotation:
   - Create a new matrix with dimensions width×height
   - For each pixel (i,j) in original, place at (width-i-1,height-j-1) in new matrix
4. For 45° rotation:
   - Create a larger matrix to accommodate rotated image
   - Calculate new dimensions based on trigonometry
   - For each pixel (x,y) in the new matrix:
     - Calculate corresponding position in original: (x×cos(45°)-y×sin(45°), x×sin(45°)+y×cos(45°))
     - If position is within original bounds, copy the pixel value
     - Otherwise, fill with background color (usually white)
5. Display the original and rotated images

*Concept:*
- Image rotation involves trigonometric transformations
- 90° and 180° rotations are special cases with simple mapping
- Non-orthogonal rotations (like 45°) require interpolation as pixels don't map exactly
- Rotations may introduce artifacts or empty spaces that need filling
- The resulting image size might change to accommodate the rotation

## Experiment 2: Creating and Manipulating Synthetic Images

### 2.a) Create Images with Alternating Lines and Perform Operations

*Algorithm for Creating Image A (Horizontal Lines):*
1. Create a blank 1024×1024 matrix filled with zeros (black)
2. For each row i from 0 to 1023:
   - If floor(i/128) is even, fill row i with ones (white)
   - Otherwise, leave as zeros (black)
3. Display Image A

*Algorithm for Creating Image B (Vertical Lines):*
1. Create a blank 1024×1024 matrix filled with zeros (black)
2. For each column j from 0 to 1023:
   - If floor(j/128) is even, fill column j with ones (white)
   - Otherwise, leave as zeros (black)
3. Display Image B

*Algorithm for Image Addition:*
1. Create a new 1024×1024 matrix C
2. For each pixel position (i,j):
   - C(i,j) = A(i,j) + B(i,j)
   - If C(i,j) > 1, set C(i,j) = 1 (clipping to maximum value)
3. Display Image C

*Algorithm for Image Subtraction:*
1. Create a new 1024×1024 matrix D
2. For each pixel position (i,j):
   - D(i,j) = A(i,j) - B(i,j)
   - If D(i,j) < 0, set D(i,j) = 0 (clipping to minimum value)
3. Display Image D

*Algorithm for Image Multiplication:*
1. Create a new 1024×1024 matrix E
2. For each pixel position (i,j):
   - E(i,j) = A(i,j) × B(i,j)
3. Display Image E

*Concept:*
- Binary images consist of only two values (0 and 1)
- Image addition combines brightness, potentially exceeding maximum value (requiring clipping)
- Image subtraction shows differences, potentially going below minimum value (requiring clipping)
- Image multiplication produces a logical AND operation for binary images
- These operations create patterns that reveal characteristics of Fourier transforms

### 2.b) Create Sinusoidal Intensity and Box Images

*Algorithm for Sinusoidal Intensity Image:*
1. Create a blank 256×1024 matrix
2. For each pixel position (i,j):
   - Calculate intensity value using sin function: 127.5 + 127.5×sin(2π×j/256)
   - This creates horizontal sinusoidal intensity variation
3. Display the resulting image

*Algorithm for Image with Black Box at Center:*
1. Create a 256×256 matrix filled with ones (white)
2. Calculate center position: center_x = 128, center_y = 128
3. Calculate box boundaries:
   - Left: center_x - 29 = 99
   - Right: center_x + 29 = 157
   - Top: center_y - 29 = 99
   - Bottom: center_y + 29 = 157
4. For each pixel position (i,j) where i is between 99 and 157 and j is between 99 and 157:
   - Set pixel value to 0 (black)
5. Display the resulting image

*Concept:*
- Synthetic images are created mathematically rather than captured
- Sinusoidal patterns are useful for testing frequency characteristics
- Precise geometric shapes like boxes help test edge detection
- These test images have known mathematical properties, making them useful for validation

## Experiment 3: Intensity Transformation Operations

### 3.a) Image Negative

*Algorithm:*
1. Load a grayscale image with intensity values in range [0,L-1], where L is maximum intensity (typically 256)
2. Create a new matrix of same dimensions
3. For each pixel with intensity value r:
   - Calculate s = L-1-r
   - Assign s to the corresponding pixel in the new matrix
4. Display the resulting negative image

*Concept:*
- Image negative reverses the intensity levels
- Dark areas become bright and vice versa
- Useful for enhancing white or gray details embedded in dark regions
- Simple transformation that requires no parameters

### 3.b) Log Transformation

*Algorithm:*
1. Load a grayscale image
2. Define the transformation parameter c (typically chosen so that maximum output is L-1)
3. Create a new matrix of same dimensions
4. For each pixel with intensity value r:
   - Calculate s = c × log(1+r)
   - Assign s to the corresponding pixel in the new matrix
5. Display the resulting transformed image

*Concept:*
- Log transformation compresses the dynamic range of high intensity values
- Expands the dynamic range of low intensity values
- Useful for enhancing details in darker regions while compressing brighter regions
- Applications include Fourier spectrum display and handling images with large dynamic range

### 3.c) Power Law Transformation

*Algorithm:*
1. Load a grayscale image
2. Define parameters c and γ (gamma)
3. Create a new matrix of same dimensions
4. For each pixel with intensity value r:
   - Calculate s = c × r^γ
   - Assign s to the corresponding pixel in the new matrix
5. Display the resulting transformed image

*Concept:*
- Power law (gamma) transformation allows flexible adjustment of intensity mapping
- γ < 1: Expands low intensity values, compresses high intensity (brightens image)
- γ > 1: Expands high intensity values, compresses low intensity (darkens image)
- Used for contrast correction in displays and cameras
- Different devices have different gamma characteristics requiring correction

### 3.d) Contrast Stretching

*Algorithm:*
1. Load a grayscale image
2. Find minimum (rmin) and maximum (rmax) intensity values in the image
3. Define desired output range [smin, smax], typically [0, L-1]
4. Create a new matrix of same dimensions
5. For each pixel with intensity value r:
   - Calculate s = (r-rmin)/(rmax-rmin) × (smax-smin) + smin
   - Assign s to the corresponding pixel in the new matrix
6. Display the resulting contrast-stretched image

*Concept:*
- Contrast stretching expands the intensity range to fill the available dynamic range
- Improves contrast by utilizing the full intensity range
- Particularly useful for low-contrast images
- Linear transformation that maps [rmin, rmax] to [smin, smax]
- Also called normalization or histogram stretching

### 3.e) Gray Level Slicing

*Algorithm:*
1. Load a grayscale image
2. Define intensity range of interest [A, B]
3. Define transformation type:
   - Type 1: Highlight range, preserve other levels
   - Type 2: Highlight range, suppress other levels
4. Create a new matrix of same dimensions
5. For each pixel with intensity value r:
   - If Type 1 and A ≤ r ≤ B, set s = max_intensity (brighten)
     Otherwise, set s = r (preserve)
   - If Type 2 and A ≤ r ≤ B, set s = max_intensity (brighten)
     Otherwise, set s = min_intensity (suppress)
6. Display the resulting sliced image

*Concept:*
- Gray level slicing highlights specific intensity ranges
- Useful for highlighting structures of interest based on intensity
- Type 1 preserves background context
- Type 2 creates a binary-like image focused on features of interest
- Applications include medical imaging and industrial inspection

## Experiment 4: Histogram Equalization

*Algorithm:*
1. Load a low contrast grayscale image
2. Calculate the histogram of the image:
   - Create an array h of size L (number of intensity levels, typically 256)
   - For each pixel with intensity r, increment h[r]
3. Calculate the normalized histogram by dividing each value by total number of pixels
4. Calculate the cumulative distribution function (CDF):
   - Initialize cdf[0] = normalized_histogram[0]
   - For i from 1 to L-1: cdf[i] = cdf[i-1] + normalized_histogram[i]
5. Create a transformation mapping using the formula:
   - For each intensity level r: mapping[r] = round((L-1) × cdf[r])
6. Create a new matrix of same dimensions
7. For each pixel with intensity r, replace with mapping[r]
8. Display the original and equalized images
9. Calculate and display the histograms of both images for comparison

*Concept:*
- Histogram equalization redistributes intensity values to achieve uniform distribution
- Enhances contrast by spreading the most frequent intensity values
- Global operation that works on the entire image
- Results in better utilization of the dynamic range
- Automatic method that doesn't require parameter tuning
- May amplify noise and can produce unnatural results in some cases

## Experiment 5: Histogram Matching/Specification

*Algorithm:*
1. Load a low contrast grayscale image A and high contrast grayscale image B
2. Calculate histograms hA and hB for both images
3. Compute normalized histograms and CDFs for both images:
   - cdfA for image A
   - cdfB for image B
4. Create a transformation mapping:
   - Initialize an array mapping of size L
   - For each intensity level r in image A:
     - Find intensity level z in image B such that |cdfB[z] - cdfA[r]| is minimized
     - Set mapping[r] = z
5. Create a new matrix of same dimensions as A
6. For each pixel in A with intensity r, replace with mapping[r]
7. Display the original image A, reference image B, and matched image
8. Calculate and display histograms of all three images for comparison

*Concept:*
- Histogram matching transforms an image to match the histogram of a reference image
- Generalizes histogram equalization (which matches to a uniform distribution)
- Useful for normalizing images for comparison or analysis
- Allows transferring contrast characteristics between images
- Applications include medical image analysis and remote sensing
- Helps standardize images captured under different conditions

## Experiment 6: Spatial Filtering for Noise Reduction

### 6.a) Linear Smoothing (Image Averaging)

*Algorithm:*
1. Load a grayscale image
2. Add salt and pepper noise:
   - For a percentage of randomly selected pixels, set value to 0 (salt) or 255 (pepper)
3. Define a linear averaging filter mask, e.g., 3×3 with all elements = 1/9
4. Apply convolution:
   - For each pixel (x,y) in the noisy image:
     - Center the mask at (x,y)
     - Calculate the sum of products of mask values and corresponding image pixel values
     - Assign this value to pixel (x,y) in the output image
5. Display the original, noisy, and filtered images

*Concept:*
- Linear smoothing replaces each pixel with weighted average of neighborhood
- Simple and computationally efficient
- Good for Gaussian noise reduction
- Reduces detail and blurs edges
- Larger masks provide more smoothing but also more blurring

### 6.b) Median Filtering

*Algorithm:*
1. Load the noise-added image from previous step
2. Define window size (typically 3×3)
3. Apply median filter:
   - For each pixel (x,y) in the noisy image:
     - Collect values of all pixels in the neighborhood defined by the window
     - Sort these values
     - Replace pixel (x,y) in the output image with the median value
4. Display the original, noisy, and median-filtered images

*Concept:*
- Median filtering is a non-linear method that replaces pixels with median of neighborhood
- Excellent for salt and pepper noise removal
- Preserves edges better than linear filtering
- More computationally intensive than linear filtering
- Very effective for removing outliers (extreme values)

### 6.c) Max Filtering

*Algorithm:*
1. Load the noise-added image from previous step
2. Define window size (typically 3×3)
3. Apply max filter:
   - For each pixel (x,y) in the noisy image:
     - Collect values of all pixels in the neighborhood defined by the window
     - Replace pixel (x,y) in the output image with the maximum value
4. Display the original, noisy, and max-filtered images

*Concept:*
- Max filtering replaces each pixel with the maximum value in its neighborhood
- Useful for finding brightest points in an image
- Tends to brighten an image and enlarge bright features
- Effective for removing 'pepper' noise (dark spots)
- Used in morphological operations (dilation)

### 6.d) Min Filtering

*Algorithm:*
1. Load the noise-added image from previous step
2. Define window size (typically 3×3)
3. Apply min filter:
   - For each pixel (x,y) in the noisy image:
     - Collect values of all pixels in the neighborhood defined by the window
     - Replace pixel (x,y) in the output image with the minimum value
4. Display the original, noisy, and min-filtered images

*Concept:*
- Min filtering replaces each pixel with the minimum value in its neighborhood
- Tends to darken an image and shrink bright features
- Effective for removing 'salt' noise (bright spots)
- Used in morphological operations (erosion)
- Can help in identifying darker regions in an image

## Experiment 7: Image Sharpening

### 7.a) Laplacian Filter

*Algorithm:*
1. Load a grayscale image
2. Define Laplacian filter mask, e.g.:
   
   [0  1  0]
   [1 -4  1]
   [0  1  0]
   
3. Apply convolution with the Laplacian mask:
   - For each pixel (x,y) in the input image:
     - Center the mask at (x,y)
     - Calculate the sum of products of mask values and corresponding image pixel values
     - Store this value as L(x,y)
4. Enhance the original image by subtracting the Laplacian:
   - For each pixel (x,y): Output(x,y) = Input(x,y) - L(x,y)
5. Display the original, Laplacian, and sharpened images

*Concept:*
- Laplacian is a second-order derivative operator that detects edges in all directions
- Emphasizes regions of rapid intensity change and de-emphasizes areas with slowly varying intensity
- Produces zero output for constant intensity regions
- When subtracted from original, sharpens edges and enhances fine details
- Sensitive to noise due to derivative nature

### 7.b) Filtering Using Composite Mask

*Algorithm:*
1. Load a grayscale image
2. Define a composite Laplacian mask directly incorporating subtraction, e.g.:
   
   [-1 -1 -1]
   [-1  9 -1]
   [-1 -1 -1]
   
3. Apply convolution with this mask:
   - For each pixel (x,y) in the input image:
     - Center the mask at (x,y)
     - Calculate the sum of products of mask values and corresponding image pixel values
     - Assign this value to pixel (x,y) in the output image
4. Display the original and sharpened images

*Concept:*
- Composite masks combine Laplacian and original image in one operation
- More computationally efficient than separate convolution and subtraction
- Different weights can be used to control sharpening strength
- Center value > sum of all other values results in sharpening
- Reduces number of steps in processing pipeline

### 7.c) Unsharp Masking

*Algorithm:*
1. Load a grayscale image
2. Create a blurred version by applying a Gaussian filter
3. Subtract the blurred image from the original to get the mask:
   - mask = original - blurred
4. Add a weighted version of the mask to the original:
   - sharpened = original + k × mask
   - Where k is a scaling factor controlling sharpening strength (typically 0.5 to 2)
5. Display the original, mask, and sharpened images

*Concept:*
- Unsharp masking creates a mask by subtracting blurred version from original
- This mask contains edge information and high-frequency details
- Adding mask to original enhances edges and details
- Name comes from traditional photography technique
- Controls sharpness without introducing significant noise
- Widely used in publishing and photography

### 7.d) High Boost Filtering

*Algorithm:*
1. Load a grayscale image
2. Create a blurred version by applying a Gaussian filter
3. Calculate high-boost filtered image:
   - highboost = A × original - blurred
   - Where A is a boosting factor > 1
4. Display the original, blurred, and high-boost filtered images

*Concept:*
- High boost filtering is a generalization of unsharp masking
- When A = 1, equivalent to unsharp masking
- When A > 1, retains some of the low-frequency components
- Provides more control over the degree of sharpening and original content
- Higher values of A produce more pronounced sharpening
- Useful when both detail enhancement and overall contrast increase are needed

### 7.e) First Order Derivative Operators (Sobel and Prewitt)

*Algorithm for Sobel:*
1. Load a grayscale image
2. Define Sobel masks for x and y directions:
   
   Gx = [-1 0 1]    Gy = [ 1  2  1]
        [-2 0 2]         [ 0  0  0]
        [-1 0 1]         [-1 -2 -1]
   
3. Apply convolution with both masks:
   - For each pixel (x,y):
     - Calculate Gx(x,y) using the x-direction mask
     - Calculate Gy(x,y) using the y-direction mask
4. Calculate gradient magnitude:
   - G(x,y) = sqrt(Gx(x,y)² + Gy(x,y)²)
   - Or approximate as |Gx(x,y)| + |Gy(x,y)|
5. Display the original image and gradient magnitude image

*Algorithm for Prewitt:*
1. Load a grayscale image
2. Define Prewitt masks for x and y directions:
   
   Gx = [-1 0 1]    Gy = [ 1  1  1]
        [-1 0 1]         [ 0  0  0]
        [-1 0 1]         [-1 -1 -1]
   
3. Apply convolution with both masks
4. Calculate gradient magnitude as before
5. Display the original image and gradient magnitude image

*Concept:*
- First-order derivatives highlight intensity changes in specific directions
- Sobel operator includes weighting to reduce noise sensitivity
- Prewitt operator uses uniform weighting
- Both emphasize vertical and horizontal edges
- Gradient magnitude combines information from both directions
- Can also determine edge direction using arctan(Gy/Gx)
- Useful for edge detection, feature extraction, and boundary identification

## Experiment 8: Fourier Transform

*Algorithm:*
1. Load images A and B from Experiment 2 (horizontal and vertical line patterns)
2. Apply 2D Fast Fourier Transform (FFT) to each image:
   - Compute F(u,v) = FFT[f(x,y)]
3. Compute magnitude spectrum:
   - |F(u,v)| = sqrt(Real²(u,v) + Imaginary²(u,v))
4. Compute log-transformed magnitude for better visualization:
   - D(u,v) = c × log(1 + |F(u,v)|)
5. Shift zero-frequency component to center of spectrum
6. Display original images and their Fourier spectra

*Concept:*
- Fourier transform decomposes an image into sinusoidal components
- Horizontal lines in spatial domain produce vertical pattern in frequency domain
- Vertical lines in spatial domain produce horizontal pattern in frequency domain
- Regular patterns have concentrated energy in frequency domain
- Low frequencies (near center) represent gradual intensity changes
- High frequencies (away from center) represent rapid changes (edges, noise)
- Magnitude spectrum shows strength of different frequency components
- Phase spectrum (not visualized here) contains structural information

## Experiment 9: Wiener Filtering for Gaussian Noise

*Algorithm:*
1. Load a grayscale image
2. Add Gaussian noise:
   - For each pixel, add a random value from Gaussian distribution
3. Apply 2D FFT to the noisy image
4. Compute Wiener filter in frequency domain:
   - H(u,v) = 1 / (1 + NSR / |F(u,v)|²)
   - Where NSR is noise-to-signal ratio
   - F(u,v) is the estimated power spectrum of the uncorrupted image
5. Apply the filter to the Fourier transform of noisy image:
   - G(u,v) = H(u,v) × F_noisy(u,v)
6. Apply inverse FFT to get the restored image
7. Display original, noisy, and restored images

*Concept:*
- Wiener filter is an optimal filter for removing Gaussian noise
- Minimizes mean square error between original and restored image
- Frequency-domain filter that adapts to image content
- Uses statistical properties of noise and image
- More effective than spatial filters for Gaussian noise
- Requires estimation of noise-to-signal ratio
- Can preserve edges better than simple smoothing
- Works well when noise characteristics are known

# Digital Image Processing Quick Reference Cheat Sheet

## Key Concepts Summary

### Basic Operations (Experiment 1)
- *Color Planes*: RGB images have 3 channels - extract them to see distribution of each color
- *Grayscale Conversion*: 0.299R + 0.587G + 0.114B (weighted average based on human perception)
- *Binary Conversion*: Apply threshold to grayscale image (pixel > threshold → white, else → black)
- *Resize*: Reduce dimensions by sampling pixels or using interpolation methods
- *Rotation*: Uses coordinate mapping with trigonometric functions

### Synthetic Images (Experiment 2)
- *Image Math Operations*:
  - Addition: Brightens image, may need clipping to handle overflow
  - Subtraction: Shows differences, needs clipping for underflow
  - Multiplication: Performs logical AND for binary images

### Intensity Transformations (Experiment 3)
- *Negative*: s = L-1-r (reverses intensity, L is max intensity value)
- *Log*: s = c×log(1+r) (enhances dark regions, compresses bright ones)
- *Power Law*: s = c×r^γ (γ<1: brightens, γ>1: darkens)
- *Contrast Stretching*: Maps [rmin,rmax] → [0,L-1] to improve contrast
- *Level Slicing*: Highlights specific intensity range of interest

### Histogram Operations (Experiments 4-5)
- *Histogram*: Counts of pixels at each intensity level
- *Histogram Equalization*: Makes intensity distribution uniform for better contrast
- *Histogram Matching*: Makes image's histogram match a reference image

### Noise Reduction (Experiment 6)
- *Salt & Pepper Noise*: Random extreme values (black/white pixels)
- *Linear Smoothing*: Simple averaging of neighborhood (blurs edges)
- *Median Filter*: Replaces with median of neighborhood (preserves edges, good for salt & pepper)
- *Max Filter*: Replaces with max value (removes dark noise, brightens image)
- *Min Filter*: Replaces with min value (removes bright noise, darkens image)

### Sharpening (Experiment 7)
- *Laplacian*: Second derivative operator that enhances edges in all directions
- *Composite Mask*: Combines Laplacian and original in one operation
- *Unsharp Masking*: original + k×(original-blurred) to enhance edges
- *High Boost*: A×original - blurred (A>1 for enhanced sharpening)
- *Edge Detection Operators*:
  - Sobel: Uses weighted masks for gradient calculation
  - Prewitt: Uses uniform weighting masks for gradient

### Frequency Domain (Experiments 8-9)
- *Fourier Transform*: Decomposes image into frequency components
- *Regular patterns* in space → concentrated energy points in frequency domain
- *Low frequencies* (center) → gradual changes
- *High frequencies* (periphery) → edges, details, noise
- *Wiener Filter*: Optimal for Gaussian noise removal in frequency domain

## Visual Memory Aids

### Transformation Functions

Image Negative:  ↗
                /
               /
              /
             /
            ↙
   0 --------- 255 (r)

Log:         ↗
            /
           /
          ↗
         ↗
        ↗
   0 --------- 255 (r)

Power (γ<1): ↗↗
            /
           ↗
          ↗
         /
        /
   0 --------- 255 (r)

Power (γ>1):    ↗
                /
               /
              /
             ↗
            ↗↗
   0 --------- 255 (r)


### Filter Masks


Averaging 3×3:    Laplacian:       Sobel (Gx):      Sobel (Gy):
[1/9 1/9 1/9]    [ 0  1  0]      [-1  0  1]      [ 1  2  1]
[1/9 1/9 1/9]    [ 1 -4  1]      [-2  0  2]      [ 0  0  0]
[1/9 1/9 1/9]    [ 0  1  0]      [-1  0  1]      [-1 -2 -1]

Composite Sharpening:
[-1 -1 -1]
[-1  9 -1]
[-1 -1 -1]


### Process Relationships

- *Blurring* ↔ *Sharpening* (Opposites)
- *Spatial Domain* ↔ *Frequency Domain* (Fourier Transform)
- *RGB* → *Grayscale* → *Binary* (Progressive reduction)
- *Histogram Equalization* is a special case of *Histogram Matching* (to uniform distribution)
- *Unsharp Masking* is a special case of *High Boost* (when A=1)

### Quick Memory Tips
- Log transform for *dark detail = **d*ynamic range compression
- γ<1 = *Less than 1 = **L*ightens image
- γ>1 = *More than 1 = **M*ore darkness
- Salt and Pepper: *Median filter = **M*agic solution
- Edges = High frequencies = Periphery in Fourier spectrum
- Smooth areas = Low frequencies = Center in Fourier spectrum
