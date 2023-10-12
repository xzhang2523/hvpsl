for seed in 0 1 2;
    do
    for decompose in epo;
        do    
        for problem in RE21 RE24 lqr2 RE37 lqr3 zdt1 zdt2 vlmop1 vlmop2;
            do python psl_main.py --decompose $decompose --n-iter 200 --problem-name $problem --seed $seed
        done
    done
done



sleep 100