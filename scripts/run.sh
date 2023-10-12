for i in {0..30}
do
  python3 split_sepsis_cohort.py
  python3 train_model.py 
done
