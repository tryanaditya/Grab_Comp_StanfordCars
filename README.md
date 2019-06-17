## Stanford Cars Dataset Classification

For more detail explanation :
https://link.to.ppt/

Train model on stanford cars dataset

```
python classify.py train dataset --batch-size 64 --pretrained 'ímagenet'

```
Test saved model on stanford cars dataset

```
python classify.py test dataset --batch-size 64 --resume saved_model/model_best.pth.tar --pretrained imagenet

```
Final Result
| Model         | Accuracy  |
|------------------------|---------------|
| DLA-102                | 93.272        |