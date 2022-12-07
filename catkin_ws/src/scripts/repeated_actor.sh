while true
	do
		python3 ../sac_discrete/src/agent/imitation/actor.py --aid $2 --port $3 --drive_mode $1 --env_mode server --lr $4
		sleep 5
	done
