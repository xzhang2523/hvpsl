



for decompose in epo
  do
  for probname in lqr2 lqr3
    do
    for seed in 0 1 2 3 4
        do
            python ./psl_main.py --problem-name $probname --n-iter 100 --seed $seed --use-plot N --decompose $decompose
    done
  done
done





sleep 100