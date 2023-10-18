for problem in zdt1 zdt2 vlmop1 vlmop2 RE21 RE24 lqr2 RE37 lqr3;  
    do python merge_table.py --problem-name $problem
done 

python merge_all.py
python -m csv2latex


sleep 100
