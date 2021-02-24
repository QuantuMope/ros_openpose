#!/usr/bin/env python

# import modules
import sys
import rospy
import argparse
import message_filters
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from ros_openpose.msg import Frame
from sensor_msgs.msg import Image, CameraInfo


# Import Openpose (Ubuntu)
py_openpose_path = rospy.get_param("~py_openpose_path")
try:
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu)
    sys.path.append(py_openpose_path)
    from openpose import pyopenpose as op
except ImportError as e:
    rospy.logerr('OpenPose library could not be found. '
                 'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


class rosOpenPose:
    def __init__(self, frame_id, pub_topic, color_topic, depth_topic, cam_info_topic, op_wrapper):
        rospy.sleep(6.0)  # so light levels in image have time to stabilize
        image_sub = message_filters.Subscriber(color_topic, Image)
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 0.01)
        self.ts.registerCallback(self.callback)

        self.pub = rospy.Publisher(pub_topic, Frame, queue_size=10)

        self.frame_id = frame_id

        self.bridge = CvBridge()

        self.op_wrapper = op_wrapper

        # This subscriber is run only once to populate necessary K matrix values.
        self.info_sub = rospy.Subscriber(cam_info_topic, CameraInfo, self.get_info_callback)
        self.fx = False
        self.fy = False
        self.cx = False
        self.cy = False

        self.depth = None

        """ OpenPose skeleton dictionary
        {0, "Nose"}, {13, "LKnee"}
        {1, "Neck"}, {14, "LAnkle"}
        {2, "RShoulder"}, {15, "REye"}
        {3, "RElbow"}, {16, "LEye"}
        {4, "RWrist"}, {17, "REar"}
        {5, "LShoulder"}, {18, "LEar"}
        {6, "LElbow"}, {19, "LBigToe"}
        {7, "LWrist"}, {20, "LSmallToe"}
        {8, "MidHip"}, {21, "LHeel"}
        {9, "RHip"}, {22, "RBigToe"}
        {10, "RKnee"}, {23, "RSmallToe"}
        {11, "RAnkle"}, {24, "RHeel"}
        {12, "LHip"}, {25, "Background"}
        """

    def get_info_callback(self, cam_info):
        self.fx = cam_info.K[0]
        self.cx = cam_info.K[2]
        self.fy = cam_info.K[4]
        self.cy = cam_info.K[5]
        self.info_sub.unregister()

    def process_depth(self, U, V, XYZ):
        num_persons, body_part_count = U.shape
        for i in range(num_persons):
            for j in range(body_part_count):
                XYZ[i, j, 2] = self.depth[V[i, j], U[i, j]]
        XYZ[:, :, 2] /= 1000  # convert to meters

    def compute_3d_vectorized(self, U, V, XYZ):
        Z = XYZ[:, :, 2]
        XYZ[:, :, 0] = (Z / self.fx) * (U - self.cx)
        XYZ[:, :, 1] = (Z / self.fy) * (V - self.cy)

    def callback(self, ros_image, ros_depth):
        # Don't process if we have not obtained K matrix yet
        if not (self.fx and self.cx and self.fy and self.cy):
            return

        # Construct a frame with current time !before! pushing to OpenPose
        fr = Frame()
        fr.header.frame_id = self.frame_id
        fr.header.stamp = rospy.Time.now()

        # Convert images to cv2 matrices
        image = None
        try:
            image = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
            self.depth = self.bridge.imgmsg_to_cv2(ros_depth, "32FC1")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

        # Push data to OpenPose and block while waiting for results
        datum = op.Datum()
        datum.cvInputData = image
        self.op_wrapper.emplaceAndPop([datum])

        pose_kp = datum.poseKeypoints
        lhand_kp = datum.handKeypoints[0]
        rhand_kp = datum.handKeypoints[1]

        # Return if we detect no one in the frame
        if pose_kp.shape == (): return
        num_persons = pose_kp.shape[0]
        body_part_count = pose_kp.shape[1]

        # Check to see if hands were detected
        lhand_detected = False
        rhand_detected = False
        hand_part_count = 0
        if lhand_kp.shape != ():
            lhand_detected = True
            hand_part_count = lhand_kp.shape[1]
        if rhand_kp.shape != ():
            rhand_detected = True
            hand_part_count = rhand_kp.shape[1]

        # Construct Frame message
        fr.persons.resize(num_persons)
        try:
            # Create views (no copies made, so this remains efficient)
            U = pose_kp[:, :, 0]
            V = pose_kp[:, :, 1]

            b_XYZ = np.zeros((num_persons, body_part_count, 3), dtype=np.float32)

            # Populate Z vector in XYZ matrix with proper depth values
            self.process_depth(U, V, b_XYZ)

            # Perform vectorized 3D computation for body keypoints
            # Populate X and Y vectors in XYZ matrix
            self.compute_3d_vectorized(U, V, b_XYZ)

            # Perform the vectorized operation for left hand
            if lhand_detected:
                U = lhand_kp[:, :, 0]
                V = lhand_kp[:, :, 1]
                lh_XYZ = np.zeros((num_persons, hand_part_count, 3), dtype=np.float32)
                self.process_depth(U, V, lh_XYZ)
                self.compute_3d_vectorized(U, V, lh_XYZ)

            # Do same for right hand
            if rhand_detected:
                U = rhand_kp[:, :, 0]
                V = rhand_kp[:, :, 1]
                rh_XYZ = np.zeros((num_persons, hand_part_count, 3), dtype=np.float32)
                self.process_depth(U, V, rh_XYZ)
                self.compute_3d_vectorized(U, V, rh_XYZ)

            for person in range(num_persons):
                fr.persons[person].bodyParts.resize(body_part_count)
                fr.persons[person].leftHandParts.resize(hand_part_count)
                fr.persons[person].rightHandParts.resize(hand_part_count)

                # Construct detected_hands iterable for later for loop
                detected_hands = []
                if lhand_detected:
                    detected_hands.append((lhand_kp, fr.persons[person].leftHandParts, lh_XYZ))
                if rhand_detected:
                    detected_hands.append((rhand_kp, fr.persons[person].rightHandParts, rh_XYZ))

                # Process body key points
                for bp in range(body_part_count):
                    u, v, s = pose_kp[person, bp]
                    x, y, z = b_XYZ[person, bp]
                    arr = fr.persons[person].bodyParts[bp]
                    arr.pixel.x = u
                    arr.pixel.y = v
                    arr.score = s
                    arr.point.x = x
                    arr.point.y = y
                    arr.point.z = z

                # Process left and right hands
                for hp in range(hand_part_count):
                    for kp, harr, h_XYZ in detected_hands:
                        u, v, s = kp[person, hp]
                        x, y, z = h_XYZ[person, hp]
                        arr = harr[hp]
                        arr.pixel.x = u
                        arr.pixel.y = v
                        arr.score = s
                        arr.point.x = x
                        arr.point.y = y
                        arr.point.z = z

        except IndexError:
            return

        self.pub.publish(fr)


def main():
    rospy.init_node('ros_openpose')
    frame_id = rospy.get_param("~frame_id")
    pub_topic = rospy.get_param("~pub_topic")
    color_topic = rospy.get_param("~color_topic")
    depth_topic = rospy.get_param("~depth_topic")
    cam_info_topic = rospy.get_param("~cam_info_topic")

    try:
        # Flags, refer to include/openpose/flags.hpp for more parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-display", action="store_true", help="Disable display.")
        parser.add_argument("--model_folder", action="store", type=str, required=True, help="Path to openpose models.")
        args = parser.parse_known_args()

        # Custom Params
        params = vars(args)
        # Can manually set params like this as well
        # params["model_folder"] = "/home/asjchoi/Programs/openpose-1.6.0/models"

        # Any more obscure flags can be found through this for loop
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in params:  params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in params: params[key] = next_item

        # Starting OpenPose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        op_wrapper.start()

        # Start ros wrapper
        rop = rosOpenPose(frame_id, pub_topic, color_topic, depth_topic, cam_info_topic, op_wrapper)
        rospy.spin()

    except Exception as e:
        print(e)
        sys.exit(-1)


if __name__ == "__main__":
    main()