dir=$1
str=$(find $dir -type f -name policy_cp_* -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/joint_pomdp_labeller_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/joint_pomdp_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/lets-drive_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
# str=$(find /home/panpan/driving_data/result/lets-drive-zero_mode/ -type f -exec stat -c '%Y %n' {} \; | sort -nr | awk 'NR==1,NR==1 {print $2}')
echo "Str: $str"
file=(${str// / })
echo "Latest policy_cp: ${file[0]}" 
vim "${file[0]}"

