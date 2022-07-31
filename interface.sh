
for seed in 5 8 47 94 106; do
    python main.py -m dummy -s ${seed}
    for budget in {1..10}; do
        for length in {3..12..3}; do
            for frequency in {3..30..3}; do
                if [ ${frequency} -ge ${length} ]; then # otherwise the attackers require more account
                    python main.py -m random -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m center -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m remote -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m highest -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m lowest -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m degree_high -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m degree_low -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m betweeness_high -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m betweeness_low -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                    python main.py -m ILP -b ${budget} -l ${length} -f ${frequency} -s ${seed}
                fi
            done
        done
    done
done
