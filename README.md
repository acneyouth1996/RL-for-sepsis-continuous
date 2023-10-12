# RL-for-sepsis-continuous

## Note

We are currently working on improving and cleaning up the code and publish a more updated version of it.

## Initial data extraction and preprocess


```bash
python3 preprocess.py
```

This step extract relavent features and perform some initial preprocess, original code is from
https://github.com/microsoft/mimic_sepsis


## Create and split sepsis cohort 

```bash
python3 sepsis_cohort.py
```

and 

```bash
python3 split_sepsis_cohort.py
```

This will create files train_set_tuples/val_set_tuples/test_set_tuples


## Policy learning

```bash
python3 train_model.py
```

## To replicate the results from our paper

```bash
sh run.sh
```





