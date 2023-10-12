for decompose in hv1 hv2 ls tche epo;
    do    
    for problem in vlmop2;
        do python psl_main.py --decompose $decompose --n-iter 1000 --problem-name $problem
    done
done






sleep 100
# for problem in zdt1 zdt2 RE21 RE24 lqr2;
#     do    
#     for decompose in hv1 hv2 ls tche;
#         do python psl_main.py --decompose $decompose --n-iter 1000 --problem-name $problem
#     done
# done



# for problem in zdt1 zdt2 RE21 RE24 RE37 dtlz2 RE37 lqr2;  
#     do    
#     for decompose in epo;
#         do python psl_main.py --decompose $decompose --n-iter 100 --problem-name $problem
#     done
# done 




# all problem
# zdt1 zdt2 RE21 RE24 RE37 dtlz2 RE37 lqr2