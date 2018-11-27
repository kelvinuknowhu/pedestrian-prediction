# Mapillary Vistas Dataset 
Welcome to the Mapillary Vistas Dataset (Research edition, v1.1)!
The public set comprises 20,000 images, out of which 18,000 shall be used for training and the remaining 2,000 for validation. The official test set now contains 5,000 RGB images, see changelog below. We provide pixel-wise labels based on polygon annotations for 66 object classes, where 37 are annotated in an instance-specific manner (i.e. individual instances are labeled separately). 

The folder structures contain raw RGB images in `{training,validation}/images`, class-specific labels for semantic segmentation in `{training,validation}/labels` (8-bit with color-palette), instance-specific annotations in `{training,validation}/instances` (16-bit) and panoptic annotations in `{training,validation}/panoptic` (24-bit RGB images). Please run 'python demo.py' from the extracted folder to get an idea about how to access label information and for mappings between label IDs and category names.

If you requested access to this dataset for your work as an employee of a company, you and the company will be bound by the license terms. Downloading the dataset implies your acceptance and the company’s acceptance of the terms. By downloading the dataset, you and the company certify that you have understood and accepted the definition of Research Purpose and the restrictions against commercial use as defined in the Mapillary Vistas Dataset Research Use License. A copy of the license terms is part of the downloaded file.

Please cite the following paper if you find Mapillary Vistas helpful for your work:

```
@InProceedings{MVD2017,
title=    {The Mapillary Vistas Dataset for Semantic Understanding of Street Scenes},
author=   {Neuhold, Gerhard and Ollmann, Tobias and Rota Bul\`o, Samuel and Kontschieder, Peter},
booktitle={International Conference on Computer Vision (ICCV)},
year=     {2017},
url=      {https://www.mapillary.com/dataset/vistas}
}
```


Mapillary Research, June 26, 2018 (https://research.mapillary.com)
