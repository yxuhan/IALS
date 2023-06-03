# Document for DT-Metric
Here we provide an example to compute DT metric with primal attribute as gender and condition attribute as smiling.
Specifically, our goal is to change the gender from female to male while keeping the smiling face.

To compute the FULL DT score like the one in our paper, you need to run the code many times with changing primal-condition pair and editing directions:
* Primal-condition pairs means the combination of `attr1` and `attr2`.
* Editing directions means the sign of step, e.g. 0.1 means female to male while -0.1 means male to female.

## Step1: Dump Images

```
python dt_score/dump_image.py \
    --n_images 5000 \
    --save_root workspace/gender_exp \
    --lambda1 0. \
    --lambda2 0.75 \
    --step 0.1 \
    --n_steps 20 \
    --attr1 male \
    --attr2 smiling
```

## Step2: Compute DT Scores

```
python dt_score/compute_score.py \
    --save_root workspace/gender_exp \
    --dt_save_path dt_curve.png \
    --step 0.1 \
    --attr1 male \
    --attr2 smiling
```

## NOTE
* The DT-Score computed from this code maybe different from the one we reported in our paper due to the random seed and (maybe) the difference of pytorch and/or other libs version.
