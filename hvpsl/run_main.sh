
# all_problems 2-obj: zdt1 zdt2 vlmop1 vlmop2 RE21 RE24 lqr2
# all_problems 3-obj: RE37 lqr3


for decompose in epo
  do
#  for probname in zdt1 zdt2 vlmop1 vlmop2 RE21 RE24 lqr2 RE37 lqr3
  for probname in lqr2 lqr3
    do
  for seed in 0 1 2 3 4
      do
          python ./hvpsl/psl_main.py --problem-name $probname --n-iter 100 --seed $seed --use-plot N --decompose $decompose
      done
  done
done




#for decompose in tche
#  do
#    for name in zdt1 zdt2 vlmop1 vlmop1 RE21 RE24 lqr2
#    do
#        python ./hvpsl/psl_main.py --problem-name $name --n-iter 5 --decompose $decompose --use-plot N
#    done
#done



sleep 100