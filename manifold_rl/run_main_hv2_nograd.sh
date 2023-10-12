


for seed in 0;
    do
    for decompose in hv2nograd;
        do    
        for problem in RE21 zdt1 RE37 RE24 zdt2 vlmop1 vlmop2 lqr2 lqr3;
            do python psl_main.py --decompose $decompose --n-iter 1000 --problem-name $problem
        done
    done
done




sleep 100
