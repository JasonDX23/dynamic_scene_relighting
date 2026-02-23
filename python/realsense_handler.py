import numpy as np
import freenect
import cv2
import logging

def to_camera_matrix(intrinsics_tuple):
    # intrinsics_tuple expected as (fx, fy, ppx, ppy)
    fx, fy, ppx, ppy = intrinsics_tuple
    return np.array([[fx,  0, ppx],
                     [ 0, fy, ppy],
                     [ 0,  0,   1]])

class RealsenseHandler():
    '''
    Modified RealsenseHandler using freenect for Kinect v1.
    Maintains original method naming and attribute structure.
    '''
    def __init__(self, resolution=(640, 480), framerate=30, decimation_magnitude=1,
                 spatial_smooth=True, temporal_smooth=True):
        
        self.w, self.h = resolution
        self.decimation_magnitude = decimation_magnitude
        self.spatial_smooth = spatial_smooth
        self.temporal_smooth = temporal_smooth
        
        # Kinect v1 Typical Intrinsics (640x480)
        # fx, fy, ppx, ppy
        self.k_params = (585.0, 585.0, 320.0, 240.0)
        
        # Scale for Kinect v1 is generally 1mm = 1 unit
        self.depth_scale = 0.001 

        # Compatibility attributes
        self.aligned_depth_K = to_camera_matrix(self.k_params)
        self.aligned_depth_inv = np.linalg.inv(self.aligned_depth_K)
        
        # Persistence for temporal smoothing
        self._last_depth = None

        # Save intrinsics as per original script
        np.savetxt("d415_intrinsics.csv", self.aligned_depth_K)

    def set_exposure(self, exposure):
        # Kinect v1 hardware/freenect exposure control is limited.
        # This is a placeholder to prevent script crashes.
        logging.info("Kinect v1 exposure set via freenect is not supported in this wrapper.")

    def get_frame(self, include_pointcloud=False, do_alignment=True):
        # 1. Fetch frames from freenect
        # DEPTH_REGISTERED aligns depth to the RGB camera's field of view
        if do_alignment:
            depth_image, _ = freenect.sync_get_depth(0, freenect.DEPTH_REGISTERED)
        else:
            depth_image, _ = freenect.sync_get_depth(0, freenect.DEPTH_11BIT)
            
        color_image, _ = freenect.sync_get_video(0, freenect.VIDEO_RGB)

        if depth_image is None or color_image is None:
            logging.error("Invalid aligned or color frame.")
            return

        # 2. Convert Color to BGR (matches Realsense/OpenCV default)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # 3. Apply simulated filters
        # Decimation
        if self.decimation_magnitude > 1:
            depth_image = depth_image[::self.decimation_magnitude, ::self.decimation_magnitude]
            # Note: This changes the resolution, which might break downstream scripts 
            # if they expect a fixed 640x480.
            
        # Spatial Smoothing (Median filter)
        if self.spatial_smooth:
            depth_image = cv2.medianBlur(depth_image.astype(np.uint16), 3)

        # Temporal Smoothing (Simple Alpha Blending)
        if self.temporal_smooth and self._last_depth is not None:
            if depth_image.shape == self._last_depth.shape:
                depth_image = cv2.addWeighted(depth_image.astype(np.float32), 0.5, 
                                            self._last_depth.astype(np.float32), 0.5, 0).astype(np.uint16)
        self._last_depth = depth_image

        # 4. Handle Pointcloud
        points = None
        if include_pointcloud:
            points = self._calculate_pc(depth_image)

        return color_image, depth_image, points

    def _calculate_pc(self, depth):
        """ Internal helper to mimic rs.pointcloud().calculate() """
        rows, cols = depth.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        
        fx, fy, ppx, ppy = self.k_params
        
        # Adjust intrinsics if decimated
        if self.decimation_magnitude > 1:
            fx /= self.decimation_magnitude
            fy /= self.decimation_magnitude
            ppx /= self.decimation_magnitude
            ppy /= self.decimation_magnitude

        z = depth.astype(float)
        x = (c - ppx) * z / fx
        y = (r - ppy) * z / fy
        
        return np.stack((x, y, z), axis=-1).reshape(-1, 3)