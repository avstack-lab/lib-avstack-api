# Downloading Data

To make use of the power of the `AVstack` API, you must download and organize some autonomous vehicle datasets. Around the internet, you will find many a number of ways AV developers have chosen to organize their AV datasets. It's important to follow the convention outlined here (or suggest a better one). AV datasets are very large, so you should plan on keeping these in a spacious location.

- **Option 1:** Use the full path to an existing download of the data.
- **Option 2:** Use the utilities in the `data` folder to download a copy of the data. This is described below.

If you chose to use an existing download *or* you choose to download the data in a non-default location, it may benefit you to create a symbolic link to the default location (see [add_custom_symlinks.sh][symlinks]). The examples will use links assuming they are in the `data` folder. Therefore, to make these work as seamlessly as possible, add the symbolic links.

Each download script takes as input the data download directory. For example, to download KITTI raw data to the folder "/data/yourname/KITTI/raw", you would run: `./download_KITTI_raw_data.sh /data/yourname". The download script will automatically append the "KITTI/raw" part to the end.

A quick explanation of each dataset and how to download is below. This assumes you've run:
```
export SAVEFOLDER=/your/path/to/data
```

## KITTI

### Object Dataset
```
./download_KITTI_ImageSets.sh $SAVEFOLDER
./download_KITTI_object_data.sh $SAVEFOLDER
```

### Raw Dataset
The raw data download will take a few hours. You can adjust the number of scenes downloaded to speed up/slow down. This will (obviously?) influence how many scenes are used for the analysis. There are only tracklet labels for up to and including `2011_09_26_drive_0091_sync`, so there is no point in using more than 38.

```
export NSCENES=38
./download_KITTI_ImageSets.sh $SAVEFOLDER  # ONLY if you didn't do it already
./download_KITTI_raw_tracklet.sh $SAVEFOLDER
./download_KITTI_raw_data.sh $SAVEFOLDER $NSCENES
```

## nuScences

Unfortunately, due to the way nuScenes hosts data, there is only a static web address for the mini dataset. All other blobs of the data [can only be accessed online][nuscenes-download]. We include the download for the mini dataset, but it is up to the end-user to create an account and download the nuScenes dataset [here][nuscenes-download]. Again, the more data blobs are downloaded, the more data can be used for the analysis.

### Mini Dataset
```
./download_nuScenes_mini.sh $SAVEFOLDER
```

## Carla Dataset

A major contribution of AVstack is to provide a simple way to create a dataset from CARLA. We went one step beyond an ego dataset and also enabled capability for collaborative sensing captures. There are two ways to get the data:

1. Generate it using the Carla simulator (faster and more fun). See [this how-to guide][generate-carla-dataset]. 
1. Download the copy we used from online (slower and less fun...but simpler). Run the following:

```
./download_CARLA_datasets.sh object-v1 $SAVEFOLDER  # for only an ego's data
./download_CARLA_datasets.sh collab-v1 $SAVEFOLDER  # for only lidar data
./download_CARLA_datasets.sh collab-v2 $SAVEFOLDER  # for camera and lidar data
```

Depending on your computation resources and internet speeds, it may be faster to just generate the data yourself with Carla. We provide a simple way to do this.


[nuscenes-download]: https://www.nuscenes.org/nuscenes#download\
[generate-carla-dataset]: https://github/com/avstack-lab/carla-sandbox/docs/how-to-guides/generate-collaborative-dataset.md
[symlinks]: https://github.com/avstack-lab/lib-avstack-api/blob/main/data/add_custom_symlinks.sh