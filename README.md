## Stanford Cars Dataset Classification

For more detail explanation :
[PDF Explanation](Tryan_Aditya.pdf)

General network architecture :
![Alt text](img/arch_image.PNG?raw=true "Title")

prerequisite :
1. Download Data Cars, make sure the folder name for data are cars_train and cars_test and place it to dataset/
2. Download best saved model from [This Link](https://drive.google.com/file/d/1-7s95JPISwVB9ZcvnQcUQ5e_0Kjn3gVG/view?usp=sharing) to saved_model/


Train model on stanford cars dataset

```
python classify.py train dataset --batch-size 64 --pretrained 'ímagenet'

```
Test saved model on stanford cars dataset

```
python classify.py test dataset --batch-size 64 --resume saved_model/best_model_1_june_1.53.pth.tar --pretrained imagenet

```

Final Result :

| Model                  | Accuracy      |
|------------------------|---------------|
| DLA-102                | 93.272        |
