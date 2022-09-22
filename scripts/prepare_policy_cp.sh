dir=$1
num=$2
mkdir -p ../sac_discrete/trained_models/0
cp $dir/critic_cp_$num ../sac_discrete/trained_models/0/critic &
cp $dir/critic_target_cp_$num ../sac_discrete/trained_models/0/critic_target &
cp $dir/value ../sac_discrete/trained_models/0/value && cp $dir/value_cp_$num ../sac_discrete/trained_models/0/value &
cp $dir/value_target ../sac_discrete/trained_models/0/value_target &
cp $dir/policy_cp_$num ../sac_discrete/trained_models/0/policy
cp $dir/alpha ../sac_discrete/trained_models/0/alpha
cp $dir/use_value ../sac_discrete/trained_models/0/use_value
echo $dir'*_cp_'$num > ../sac_discrete/trained_models/what.txt

