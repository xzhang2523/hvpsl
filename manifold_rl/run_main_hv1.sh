for seed in 0;
    do
    for decompose in hv1;
        do    
        for problem in zdt2;
            do python psl_main_hyper.py --decompose $decompose --n-iter 500 --problem-name $problem --seed $seed
        done
    done
done







sleep 100
