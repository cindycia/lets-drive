mode=$1

cd raw_images

echo "converting to video..."
ffmpeg -threads 60 -framerate 60 -i $mode'_'3111'_'frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $mode'_'3111.mp4
ffmpeg -threads 60 -framerate 60 -i $mode'_'4111'_'frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $mode'_'4111.mp4
ffmpeg -threads 60 -framerate 60 -i $mode'_'5111'_'frame%04d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p $mode'_'5111.mp4
echo "cleaning up..."
rm $mode'_3111_'*.jpg
rm $mode'_4111_'*.jpg
rm $mode'_5111_'*.jpg

