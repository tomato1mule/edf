import numpy as np
import cv2

def get_image_transform(theta, trans, pivot=(0, 0)):
    """Compute composite 2D rigid transformation matrix."""
    # Get 2D rigid transformation matrix that rotates an image by theta (in
    # radians) around pivot (in pixels) and translates by trans vector (in
    # pixels)
    pivot_t_image = np.array([[1., 0., -pivot[0]], [0., 1., -pivot[1]],
                            [0., 0., 1.]])
    image_t_pivot = np.array([[1., 0., pivot[0]], [0., 1., pivot[1]],
                            [0., 0., 1.]])
    transform = np.array([[np.cos(theta), -np.sin(theta), trans[0]],
                        [np.sin(theta), np.cos(theta), trans[1]], [0., 0., 1.]])
    return np.dot(image_t_pivot, np.dot(transform, pivot_t_image))

def get_random_image_transform_params(image_size):
    theta_sigma = 2 * np.pi / 6
    theta = np.random.normal(0, theta_sigma)

    trans_sigma = np.min(image_size) / 6
    trans = np.random.normal(0, trans_sigma, size=2)  # [x, y]
    pivot = (image_size[1] / 2, image_size[0] / 2)
    return theta, trans, pivot

def perturb(input_image, pixels, set_theta_zero=False, rim_offset = 0):
    """Data augmentation on images."""
    image_size = input_image.shape[:2]

    # Compute random rigid transform.
    while True:
        theta, trans, pivot = get_random_image_transform_params(image_size)
        if set_theta_zero:
            theta = 0.
        transform = get_image_transform(theta, trans, pivot)
        transform_params = theta, trans, pivot
        #print(trans)

        # Ensure pixels remain in the image after transform.
        is_valid = True
        new_pixels = []
        new_rounded_pixels = []
        for pixel in pixels:
            pixel = np.float32([pixel[1], pixel[0], 1.]).reshape(3, 1)

            rounded_pixel = np.int32(np.round(transform @ pixel))[:2].squeeze()
            rounded_pixel = np.flip(rounded_pixel)

            pixel = (transform @ pixel)[:2].squeeze()
            pixel = np.flip(pixel)

            in_fov_rounded = rounded_pixel[0] < image_size[0]-rim_offset and rounded_pixel[1] < image_size[1]-rim_offset
            in_fov = pixel[0] < image_size[0]-rim_offset and pixel[1] < image_size[1]-rim_offset

            is_valid = is_valid and np.all(rounded_pixel >= rim_offset) and np.all(pixel >= rim_offset) and in_fov_rounded and in_fov

            new_pixels.append(pixel)
            new_rounded_pixels.append(rounded_pixel)
        if is_valid:
            break

    # Apply rigid transform to image and pixel labels.
    input_image = cv2.warpAffine(
        input_image,
        transform[:2, :], (image_size[1], image_size[0]),
        flags=cv2.INTER_NEAREST)
    return input_image, new_pixels, new_rounded_pixels, transform_params