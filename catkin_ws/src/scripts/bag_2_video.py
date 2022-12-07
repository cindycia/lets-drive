#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os, sys
import argparse

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    try:
        """Extract a folder of images from a rosbag.
        """
        parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
        parser.add_argument("--bag", help="Input ROS bag.")
        parser.add_argument("--mode", help="Input ROS bag.")
        parser.add_argument("--port", help="Input ROS bag.")
        parser.add_argument("--output_dir", required=False, default="raw_images", help="Output directory.")
        parser.add_argument("--image_topic_front", required=False, default="/spectator_images_front", help="Image topic.")
        parser.add_argument("--image_topic_top", required=False, default="/spectator_images_top", help="Image topic.")

        args = parser.parse_args()

        print "Extract images from %s on topic %s and %s into %s" % (args.bag,
                                                              args.image_topic_top, args.image_topic_front, args.output_dir)

        bag = rosbag.Bag(args.bag, "r")
        topics = bag.get_type_and_topic_info()[1].keys()
        print(topics)
        sys.stdout.flush()
        bridge = CvBridge()
        count = 0
        for topic, msg, t in bag.read_messages(topics=[args.image_topic_top]):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv2.imwrite(os.path.join(args.output_dir, "top_{}_{}_".format(args.mode, args.port)+ ("frame%04d.jpg" % count)), cv_img)
            print "Wrote top image %i" % count
            count += 1

        # count = 0
        # for topic, msg, t in bag.read_messages(topics=[args.image_topic_front]):
        #     cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        #     cv2.imwrite(os.path.join(args.output_dir, "front_{}_{}_".format(args.mode, args.port)+ ("frame%04d.jpg" % count)), cv_img)
        #     print "Wrote front image %i" % count
        #     count += 1

        bag.close()
    except Exception as e:
        print(e)
        sys.stdout.flush()

    return

if __name__ == '__main__':
    main()
