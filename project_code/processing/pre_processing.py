'''This processing script takes the images from raw/ s3 bucket folder,

1. resizes the images (32,32) and stores them,
2. Normalize the images /255.0,
3. Handle missing images cleaned_images = [img for img in images if img is not None],
4. in an s3 bucket folder by class=??/ '''