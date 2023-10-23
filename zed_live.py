import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import cv2
import time
from pathlib import Path
import enum
import numpy as np

#It records only LEFT


# def handler(signal_received, frame):
#     cam.disable_recording()
#     cam.close()
#     sys.exit(0)

#signal(SIGINT, handler)

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

video_name='/home/joao/Zed/svo/point_1'
fps = 30

if(fps != 30):
    print('FPS NOT 30!')
    sys.exit(0)

def record():

    cam = sl.Camera()
    
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    init_params.camera_fps = fps
    
    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()

    # If the camera is static, uncomment the following line to have better performances
    positional_tracking_parameters.set_as_static = True

    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
        
    # Get image size
    image_size = cam.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    
    path_output = f'svo/{video_name}.svo'

    #sl.SVO_COMPRESSION_MODE.LOSSLESS
    recording_param = sl.RecordingParameters(path_output, sl.SVO_COMPRESSION_MODE.LOSSLESS , target_framerate=30)
    err = cam.enable_recording(recording_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit(1)
    
    runtime = sl.RuntimeParameters()
    runtime.sensing_mode = sl.SENSING_MODE.FILL
    print("SVO is Recording, use Ctrl-C to stop.")
    svo_image = sl.Mat()
    
    n_frame = 0
    print("2 Second Delay")
    time.sleep(2)
    
    while True:
        if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS :
            cam.retrieve_image(svo_image) #cam.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE)
            cv2.imshow("ZED", cv2.resize(svo_image.get_data(),
                                   (int(1920 / 1.5), int(1080 / 1.5))))

            # Copy the left image to the left side of SBS image
            svo_image_sbs_rgba[:, :, :] = svo_image.get_data()
            
            key = cv2.waitKey(delay=2)
            if key == ord('q'):
                break
                    
            

    print(f"Finished recording {n_frame-1} frames")
    return 0         
            
class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3
    LEFT_ONLY = 4

def export():
    svo_input_path = Path(f'{video_name}.svo')
    output_path = Path(f'videos/{video_name}_rgb.avi')
    
    app_type = AppType.LEFT_ONLY
    
    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height
    width_sbs = width * 2
    
    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_sbs_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    
    video_writer = cv2.VideoWriter(str(output_path),
                                       cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                       max(zed.get_camera_information().camera_configuration.fps, 25),
                                       (width, height))
    
    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()
    
    if nb_frames > fps*5:
        print("ERROR: Number of frames greater than 150, closing program!")
        sys.exit(0)
    
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            # Copy the left image to the left side of SBS image
            svo_image_sbs_rgba[:, :, :] = left_image.get_data()

            # Convert SVO image from RGBA to RGB
            ocv_image_sbs_rgb = cv2.cvtColor(svo_image_sbs_rgba, cv2.COLOR_RGBA2RGB)

            # Write the RGB image in the video
            video_writer.write(ocv_image_sbs_rgb)

            # Display progress
            progress_bar((svo_position + 1) / nb_frames * 100, 30)

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
            
    # Close the video writer
    video_writer.release()

    zed.close()
    return 0            
    
if __name__ == "__main__":
    #record()
    export()
    sys.exit(0)





