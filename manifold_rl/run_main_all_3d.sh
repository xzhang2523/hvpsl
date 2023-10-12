# for problem in lqr3;
#     do    
#     for decompose in hv2;  
#         do python psl_main.py --decompose $decompose --n-iter 1000 --problem-name $problem
#     done
# done

for problem in dtlz2;
    do
    for decompose in mtche mtchenograd;
        do python psl_main.py --decompose $decompose --n-iter 2000 --problem-name $problem
    done
done 


# for problem in zdt1 zdt2 RE21 RE24 RE37 dtlz2 RE37 lqr2;  
#     do    
#     for decompose in epo;
#         do python psl_main.py --decompose $decompose --n-iter 100 --problem-name $problem
#     done
# done 




# all problem
# zdt1 zdt2 RE21 RE24 RE37 dtlz2 RE37 lqr2

sleep 100