import imageio.v2 as imageio
import os

def make_video_from_images(
    run, 
    output_video = "voronoi_video.mp4", 
    image_prefix="voronoi_", 
    image_suffix=".png",
    subfolder="images",
    start=0,
    step=5,
    end=None,
    fps=10
):
    """
    Combines sequential images into a video.
    
    Parameters
    ----------
    image_folder : str
        Directory where images are stored.
    output_video : str
        Filename for output video (e.g. 'output.mp4').
    image_prefix : str
        Common prefix for image filenames.
    image_suffix : str
        Image file extension.
    start : int
        Starting image index.
    step : int
        Step size between images.
    end : int
        Optional: last index to include (if None, auto-detect).
    fps : int
        Frames per second of output video.
    """
    image_folder = os.path.join("data", run, subfolder)
    output_dir = f"data/{run}"
    if not os.path.isdir(image_folder):
        print(f"Folder path: {image_folder}") 
        raise FileNotFoundError(f"Image folder '{image_folder}' not found.")

    output_path = os.path.join(output_dir, output_video)
    writer = imageio.get_writer(output_path, fps=fps)

    if end is None:
        # Try to find maximum index automatically
        files = [f for f in os.listdir(image_folder) if f.startswith(image_prefix) and f.endswith(image_suffix)]
        indices = [int(f[len(image_prefix):-len(image_suffix)]) for f in files]
        end = max(indices)

    for i in range(start, end + 1, step):
        filename = os.path.join(image_folder, f"{image_prefix}{i}{image_suffix}")
        if os.path.exists(filename):
            image = imageio.imread(filename)
            writer.append_data(image)
        else:
            print(f"Warning: {filename} not found, skipping.")

    writer.close()
    print(f"Saved video as {output_video}")