#!/usr/bin/env bash
  
# Ref: https://blog.csdn.net/leida_wt/article/details/115120940

STR='https:\/\/raw.githubusercontent.com'
GHPROXY='https:\/\/ghproxy.com\/'

if [ -f /usr/lib/python3/dist-packages/rosdep2/sources_list.py.bak ]; then
        echo "PASS to EXIT!!!!!!"
        exit 0
else

        sudo cp -rvf \
                /etc/ros/rosdep/sources.list.d/20-default.list \
                /etc/ros/rosdep/sources.list.d/20-default.list.bak
        sudo cp -rvf \
                /usr/lib/python3/dist-packages/rosdistro/__init__.py \
                /usr/lib/python3/dist-packages/rosdistro/__init__.py.bak
        sudo cp -rvf \
                /usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py \
                /usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py.bak
        sudo cp -rvf \
                /usr/lib/python3/dist-packages/rosdep2/sources_list.py \
                /usr/lib/python3/dist-packages/rosdep2/sources_list.py.bak
        sudo cp -rvf \
                /usr/lib/python3/dist-packages/rosdep2/rep3.py \
                /usr/lib/python3/dist-packages/rosdep2/rep3.py.bak
        sudo cp -rvf \
                /usr/lib/python3/dist-packages/rosdistro/manifest_provider/github.py \
                /usr/lib/python3/dist-packages/rosdistro/manifest_provider/github.py.bak
        sudo cp -rvf \
                /usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py \
                /usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py.bak
fi


# 1

sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /etc/ros/rosdep/sources.list.d/20-default.list
sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /usr/lib/python3/dist-packages/rosdistro/__init__.py
sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py
sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /usr/lib/python3/dist-packages/rosdep2/sources_list.py
sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /usr/lib/python3/dist-packages/rosdep2/rep3.py
sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /usr/lib/python3/dist-packages/rosdistro/manifest_provider/github.py
sudo sed -i "s/${STR}/${GHPROXY}${STR}/g" \
        /usr/lib/python3/dist-packages/rosdep2/gbpdistro_support.py
