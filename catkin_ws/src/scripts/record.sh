flag=$1
rm -r runs_$flag
cp -r runs runs_$flag
rm -r ../sac_discrete/trained_models_$flag
cp -r ../sac_discrete/trained_models ../sac_discrete/trained_models_$flag
# rm -r ~/replay_$flag
# cp -r ~/replay ~/replay_$flag
