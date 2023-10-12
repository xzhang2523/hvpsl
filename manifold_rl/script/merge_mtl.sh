for problem in adult default;  
    do python merge_table.py --problem-name $problem
done 


python merge_all.py
python -m csv2latex