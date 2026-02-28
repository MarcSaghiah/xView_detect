# How to Install the xView Dataset

## 1. Understand the xView Dataset

The **xView dataset** is a large-scale satellite imagery dataset (WorldView-3) with bounding box annotations for **over a million objects** in **60 different classes**.  

- Official website: [xviewdataset.org](https://xviewdataset.org)  
- Challenge website: [challenge.xviewdataset.org](https://challenge.xviewdataset.org)  

> Note: The dataset is **not downloadable automatically**; you need to register or log in on the official site.

---

## 2. Where to Download

- Go to the **DIUx xView 2018 Detection Challenge page**:  
  [https://challenge.xviewdataset.org](https://challenge.xviewdataset.org)  
- Register or log in to access the dataset downloads.  
- Download the files you need:

### Images
- `train_images.zip` – training images  
- `val_images.zip` – validation images  

### Annotations
- `xView_train.geojson` – training bounding boxes  
- Validation annotations may also be provided.

> These file names are the standard ones expected by most processing scripts.

---

## 3. Extract and Organize

After downloading:

1. Create a folder for your dataset, e.g.:

```bash
data/raw/xview-dataset
```

2. Extract images:

```bash
data/xview-dataset/train_images/
data/xview-dataset/val_images/
```

3. Place annotation files in:

```bash
data/xview-dataset/train_labels/xView_train.geojson
```

---

## 4. Process Annotations

- The `xView_train.geojson` file contains **all bounding boxes in one file**.  
- Many frameworks (YOLO, COCO) require annotations **split per image**.  
- Use scripts to **split geoJSON or convert to your desired format**.

---


## 5. License

- xView dataset is under **CC BY-NC-SA 4.0**, for **non-commercial use only**.  
- Check [xviewdataset.org](https://xviewdataset.org) for full licensing details.
