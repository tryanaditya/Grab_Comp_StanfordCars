## Stanford Cars Dataset Classification using Deep Layer Aggregation

https://docs.google.com/presentation/d/1CWeE5Yh1PtpTe2GEBoU3PQ4dy_LclhT--5Xq17cKmm8/edit?usp=sharing

Train model on stanford cars dataset

```
python classify.py train dataset --batch-size 664 --pretrained 'ímagenet'

```
Test saved model on stanford cars dataset

```
python classify.py test dataset --batch-size 664 --resume saved_model/model_best.pth.tar --pretrained imagenet

```

