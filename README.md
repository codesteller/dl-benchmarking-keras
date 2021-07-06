# keras-dl-modeling

## Dataset
This work is done on a simple dataset [Kaggle Cats and Dogs](https://www.kaggle.com/c/dogs-vs-cats/data)

Raw data can be also downloaded from [here](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip). This iwll downlad a 786M ZIP archive of the raw data spilt into class "dogs" & "cats". It has 25000 images per class. 

```
curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
unzip -q kagglecatsanddogs_3367a.zip
```

## Benchmark ImageDataGenerot with TF Dataset
To run the benchmarking, change line number 78 with the different parameter and see the training time. "tf_data" with  prefetch and cache is about 3x faster compared to ImageDataGenerator. 
```
data_api = "keras_gen"         # "tf_data" or "keras_gen"
```