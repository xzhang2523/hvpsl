for niter in 500;
    do    
    for decompose in hv2;
        do python psl_main_visual.py --decompose $decompose --n-iter $niter --problem-name RE37
    done
done 
