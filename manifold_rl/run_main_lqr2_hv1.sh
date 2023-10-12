for seed in 0 1 2;
    do
    for decompose in hv1;
        do    
        for problem in lqr2;
            do python psl_main.py --decompose $decompose --n-iter 1000 --problem-name $problem --seed $seed
        done
    done
done



sleep 100