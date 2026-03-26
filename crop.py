# Crop function to crop the image to remove the plt title

from PIL import Image

def extract_crop_box(image_path, title_height):
    """
    Extracts the crop box coordinates based on the image dimensions and title height.

    Parameters:
    - image_path: str, path to the input image
    - title_height: int, height of the title area to be cropped

    Returns:
    - crop_box: tuple, (left, upper, right, lower) pixel coordinates for cropping
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # Define the crop box to remove the title area
            crop_box = (0, title_height, width, height)
            return crop_box
    except Exception as e:
        print(f"An error occurred while extracting the crop box: {e}")
        return None

def crop_image(image_path, output_path, crop_box):
    """
    Crops the image to remove the plt title.

    Parameters:
    - image_path: str, path to the input image
    - output_path: str, path to save the cropped image
    - crop_box: tuple, (left, upper, right, lower) pixel coordinates for cropping
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Crop the image using the provided crop box
            cropped_img = img.crop(crop_box)
            # Save the cropped image to the specified output path
            cropped_img.save(output_path)
            print(f"Cropped image saved to {output_path}")
    except Exception as e:
        print(f"An error occurred while cropping the image: {e}")
        
if __name__ == "__main__":
    # Example usage
    input_image_path = "report/images/guided_unet_0___zero_s-1.0.png"
    output_image_path = "report/images/guided_unet_0___zero_s-1.0_cropped.png"
    title_height = 30 

    crop_box = extract_crop_box(input_image_path, title_height)
    if crop_box:
        crop_image(input_image_path, output_image_path, crop_box)