# roscore &
mkdir -p raw_images
roslaunch bag_2_video.launch
mv ~/.ros/frame*.jpg raw_images/
cd raw_images
echo "converting to video..."
ffmpeg -framerate 60 -i frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
echo "cleaning up..."
yes | rosclean purge
pkill -9 roslaunch
pkill -9 rosmaster
pkill -9 roscore
pkill -9 rosout
rm frame*.jpg
