


for decompose in ls tche hv1 hv2
  do
    for name in zdt1 zdt2 vlmop1 vlmop2 RE21 RE24 lqr2
    do
        python ./hvpsl/psl_main.py --problem-name $name --n-iter 500 --decompose $decompose --use-plot N
    done
done

sleep 100