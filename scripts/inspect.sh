str=$(find /home/panpan/driving_data/result/lets-drive-labeller_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/joint_pomdp_labeller_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/joint_pomdp_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/lets-drive_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/lets-drive-zero_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
echo "Str: $str"
file=(${str// / })
echo "Latest log: ${file[0]}" 
vim "${file[0]}"

