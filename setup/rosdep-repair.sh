#!/usr/bin/env bash

# Ref: https://blog.csdn.net/leida_wt/article/details/115120940

STR='https:\/\/raw.githubusercontent.com'
GHPROXY='https:\/\/ghproxy.com\/'

# 1
if [ -f /usr/lib/python3/dist-packages/rosdep2/sources_list.py.bak ]; then
	echo "PASS to EXIT!!!!!!"
	exit 0
else

	sudo cp -rvf /usr/lib/python3/dist-packages/rosdep2/sources_list.py \
	             /usr/lib/python3/dist-packages/rosdep2/sources_list.py.bak

	# sudo sed -i "311s/^/        url=\"${GHPROXY}\"+url\n/" \
	#	/usr/lib/python3/dist-packages/rosdep2/sources_list.py
	
	sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
	/etc/ros/rosdep/sources.list.d/20-default.list
fi


# 2
sudo sed -i "68s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdistro/__init__.py


# 3
sudo sed -i "34s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py


# 4
sudo sed -i "64s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdep2/sources_list.py


# 5
sudo sed -i "36s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdep2/rep3.py


# 6
sudo sed -i "68s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdistro/manifest_provider/github.py

sudo sed -i "119s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdistro/manifest_provider/github.py

# 7
sudo sed -i "34s/https:\/\//${GHPROXY}/" \
	/usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py
