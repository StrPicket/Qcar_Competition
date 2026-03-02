#! /usr/bin/env python3

# Generic python packages
import time  # Time library
import numpy as np
import cv2

# ROS specific packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

'''
Secondary node to determine road signs and traffic lights based on classic
object and color detection techniques.
'''

class ObjectDetector(Node):

    def __init__(self):
      super().__init__('traffic_system_detector')

      qos = QoSProfile(
      	reliability=ReliabilityPolicy.RELIABLE,
      	history=HistoryPolicy.KEEP_LAST,
      	depth=1
      )
      self.camera_image_subscriber = self.create_subscription(Image, '/qcar_camera/rgb', self.image_callback, qos)

      #self.camera_image_subscriber = self.create_subscription(Image ,'/camera/color_image',self.image_callback, 10)

      self.motion_publisher = self.create_publisher(Bool,'motion_enable',10)
      self.motion_enable = True
      self.detection_cooldown = 10.0
      self.disable_until = 0.0
      self.light_detected = False

      self.publish_motion_flag(True)
      self.t0 = time.time()

      self.bridge = CvBridge()
      self.sign_detected = False

    def image_callback(self,msg):
      current_time = time.time()-self.t0
      delay = 0
      sign_delay = 0
      light_delay = 0
      light_detected = False
      sign_detected = False
      cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


      try:
        height, width, channel = cv_image.shape
        # Convert from BGR to HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        if not self.sign_detected and not self.light_detected:
            # send image to the sign detector to check for a sign in the scene and return
            # a delay based on what's seen
            sign_delay, sign_detected = self.sign_detector(hsv=hsv)

            if not self.light_detected:
              if sign_detected:
                delay = sign_delay


            if delay > 0.0 and not self.sign_detected:
              self.sign_detected = True
              self.disable_until= delay
              self.publish_motion_flag(False)
            else:
              self.publish_motion_flag(True)



        if self.sign_detected:
          if current_time >= self.disable_until:
            if current_time >= self.detection_cooldown:
              self.sign_detected = False
            self.publish_motion_flag(True)

      except AttributeError:
          self.get_logger().info("No image received")


    def publish_motion_flag(self, enable:bool):
       msg = Bool()
       msg.data = enable
       self.motion_publisher.publish(msg)

    def sign_detector(self, hsv):
        detected = False
        delay = 0.0
        height, width, _ =  hsv.shape

        roi = hsv[:, int(3*width/4)::]

        # Define the red color range in HSV
        lower_red1 = np.array([0,120,70])
        upper_red1 = np.array([10,255,255])
        lower_red2 = np.array([170,120,70])
        upper_red2 = np.array([180,255,255])

        mask1 = cv2.inRange(roi,lower_red1,upper_red1)
        mask2 = cv2.inRange(roi,lower_red2,upper_red2)
        mask = mask1 | mask2

        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        #find countours
        contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
          area = cv2.contourArea(cnt)
          if area < 500:
            continue

          #approximate shape
          epsilon = 0.04*cv2.arcLength(cnt,True)
          approx = cv2.approxPolyDP(cnt,epsilon,True)
          sides = len(approx)

          self.get_logger().info(f"[SIGN DEBUG] area={area:.0f} sides={sides}")
          if sides == 3:
            self.get_logger().info("[SIGNAL SISTEM]Yield! Pausing for 1.5s...")
            delay = 1.5
            self.t0 = time.time()
            detected = True
            self.detection_cooldown =10.0
        #cv2.imshow("Sign Detection Mask",mask)
        #cv2.waitKey(1)
        return delay, detected

def main():

  # Start the ROS 2 Python Client Library
  rclpy.init()

  node = ObjectDetector()
  try:
      rclpy.spin(node)
  except KeyboardInterrupt:
      pass

  rclpy.shutdown()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
