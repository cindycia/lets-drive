python clear_process.py
rm -r ~/replay &
rm -r runs/ &
rm -r ../sac_discrete/trained_models/* &
rm ../crowd_pomdp_planner/temp_policy_net_0.pt
rm ../crowd_pomdp_planner/temp_value_net_0.pt
yes | docker network prune
yes | docker container prune
sudo rm -r ~/driving_data/result/lets-drive-zero_mode
sudo rm -r ~/driving_data/result/lets-drive-labeller_mode
sudo rm -r ~/driving_data/result/lets-drive_mode
sudo rm -r ~/driving_data/result/imitation_mode
sudo rm -r ~/driving_data/result/joint_pomdp_mode
sudo rm -r ~/driving_data/result/imitation_explore_mode
